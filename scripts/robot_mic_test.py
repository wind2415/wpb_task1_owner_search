#!/usr/bin/env python3
# coding: utf-8
"""Minimal robot microphone test.

This uses the same ALSA default-capture idea as the wpb_home/xfyun voice stack,
but avoids ASR, TTS, Ollama, and task logic. If this script does not show RMS
changes when speaking, the problem is below the speech recognizer.
"""

import argparse
import audioop
import os
import signal
import subprocess
import sys
import time
import wave


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATKIN_SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OFFLINE_VOICE_DIR = os.path.join(CATKIN_SRC_DIR, "offline_voice_bridge")
DEFAULT_WHISPER_ZH_MODEL = os.path.join(OFFLINE_VOICE_DIR, "models", "whisper", "faster-whisper-small")


def build_parser():
    parser = argparse.ArgumentParser(description="Test robot microphone capture through ALSA/arecord.")
    parser.add_argument(
        "--device",
        default="default",
        help="ALSA capture device. Use default to match wpb_home; example: plughw:CARD=Generic_1,DEV=0",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--chunk-ms", type=float, default=200.0, help="RMS report interval in milliseconds")
    parser.add_argument("--threshold", type=int, default=30, help="Advisory RMS threshold for loud speech")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means run until Ctrl+C")
    parser.add_argument("--list-devices", action="store_true", help="Print arecord -l and arecord -L before capture")
    parser.add_argument("--save-wav", help="Optional wav output path for captured audio")
    parser.add_argument("--summary-interval", type=float, default=1.0)
    parser.add_argument("--no-asr", action="store_true", help="Only test microphone RMS; do not run Chinese ASR")
    parser.add_argument("--asr-window", type=float, default=4.0, help="Audio seconds per Chinese ASR attempt")
    parser.add_argument("--model-size", default=DEFAULT_WHISPER_ZH_MODEL, help="faster-whisper Chinese model path/name")
    parser.add_argument("--hf-endpoint", default="https://huggingface.co")
    parser.add_argument("--asr-device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--no-vad", action="store_true", help="Disable faster-whisper VAD for fixed short windows")
    parser.add_argument("--vad-filter", dest="no_vad", action="store_false", help="Enable faster-whisper VAD")
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    parser.set_defaults(no_vad=True)
    return parser


def run_command(title, cmd):
    print("\n=== %s ===" % title)
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, text=True)
        print(proc.stdout.strip() or "<empty>")
    except OSError as exc:
        print("ERROR: %s" % exc)


def open_wave_writer(path, sample_rate, channels):
    if not path:
        return None
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    wav_file = wave.open(path, "wb")
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    return wav_file


def terminate_process(proc):
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=1.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def write_wav(path, raw, sample_rate, channels):
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw)


def load_asr_model(args):
    if args.no_asr:
        return None
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run offline_voice_bridge/tools/setup_offline_voice.sh first. %s"
            % exc
        )

    print("Loading Chinese ASR model: %s (%s/%s)" % (args.model_size, args.asr_device, args.compute_type))
    start = time.time()
    model = WhisperModel(args.model_size, device=args.asr_device, compute_type=args.compute_type)
    print("Chinese ASR model loaded in %.2fs" % (time.time() - start))
    return model


def transcribe_raw_audio(args, model, raw, round_index):
    if model is None:
        return ""
    wav_path = "/dev/shm/robot_mic_test_asr_%03d.wav" % round_index
    write_wav(wav_path, raw, args.sample_rate, args.channels)
    start = time.time()
    segments, _ = model.transcribe(
        wav_path,
        language=args.language,
        vad_filter=not args.no_vad,
        beam_size=args.beam_size,
        condition_on_previous_text=False,
        no_speech_threshold=args.no_speech_threshold,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    try:
        os.unlink(wav_path)
    except OSError:
        pass
    print("识别文本: %s  (ASR %.2fs)" % (text or "<空>", time.time() - start))
    return text


def main():
    args = build_parser().parse_args()
    bytes_per_sample = 2
    chunk_seconds = max(0.02, args.chunk_ms / 1000.0)
    chunk_bytes = int(args.sample_rate * args.channels * bytes_per_sample * chunk_seconds)
    chunk_bytes = max(bytes_per_sample * args.channels, chunk_bytes)
    asr_window_bytes = int(max(chunk_seconds, args.asr_window) * args.sample_rate * args.channels * bytes_per_sample)

    if args.list_devices:
        run_command("arecord -l", ["arecord", "-l"])
        run_command("arecord -L", ["arecord", "-L"])

    cmd = [
        "arecord",
        "-q",
        "-D",
        args.device,
        "-f",
        "S16_LE",
        "-r",
        str(args.sample_rate),
        "-c",
        str(args.channels),
        "-t",
        "raw",
    ]
    print("\nStarting microphone capture:")
    print("  %s" % " ".join(cmd))
    print("Speak near the robot microphone. RMS above %d is printed as VOICE." % args.threshold)
    if args.no_asr:
        print("Chinese ASR is disabled by --no-asr.")
    else:
        print("Chinese ASR window: %.1fs, model: %s" % (args.asr_window, args.model_size))
    print("Press Ctrl+C to stop.\n")

    start_time = time.time()
    deadline = start_time + args.seconds if args.seconds and args.seconds > 0 else None
    last_summary = start_time
    summary_samples = []
    total_chunks = 0
    voice_chunks = 0
    max_rms = 0
    max_peak = 0
    asr_round = 1
    asr_buffer = bytearray()
    asr_window_voice = False
    wav_file = None
    proc = None

    try:
        asr_model = load_asr_model(args)
        wav_file = open_wave_writer(args.save_wav, args.sample_rate, args.channels)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while True:
            now = time.time()
            if deadline is not None and now >= deadline:
                break

            raw = proc.stdout.read(chunk_bytes)
            if not raw:
                stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                print("ERROR: arecord stopped. %s" % (stderr.strip() or "no stderr"))
                return 1

            if wav_file is not None:
                wav_file.writeframes(raw)

            rms = audioop.rms(raw, bytes_per_sample)
            peak = audioop.max(raw, bytes_per_sample)
            max_rms = max(max_rms, rms)
            max_peak = max(max_peak, peak)
            total_chunks += 1
            if rms >= args.threshold:
                voice_chunks += 1
                state = "VOICE"
                asr_window_voice = True
            else:
                state = "quiet"

            if asr_model is not None:
                asr_buffer.extend(raw)
                if len(asr_buffer) >= asr_window_bytes:
                    window = bytes(asr_buffer[:asr_window_bytes])
                    del asr_buffer[:asr_window_bytes]
                    if asr_window_voice:
                        print("\n--- 中文识别窗口 %d ---" % asr_round)
                        transcribe_raw_audio(args, asr_model, window, asr_round)
                    else:
                        print("\n--- 中文识别窗口 %d: 音量低，跳过 ASR ---" % asr_round)
                    asr_round += 1
                    asr_window_voice = False

            summary_samples.append(rms)
            if now - last_summary >= max(0.2, args.summary_interval):
                avg = sum(summary_samples) / float(len(summary_samples)) if summary_samples else 0.0
                recent_max = max(summary_samples) if summary_samples else 0
                print(
                    "t=%5.1fs  rms=%5d  peak=%5d  avg=%6.1f  recent_max=%5d  max_rms=%5d  max_peak=%5d  %s"
                    % (now - start_time, rms, peak, avg, recent_max, max_rms, max_peak, state)
                )
                summary_samples = []
                last_summary = now

        elapsed = max(0.001, time.time() - start_time)
        print("\nDone. chunks=%d voice_chunks=%d max_rms=%d max_peak=%d elapsed=%.1fs voice_ratio=%.2f" % (
            total_chunks,
            voice_chunks,
            max_rms,
            max_peak,
            elapsed,
            voice_chunks / float(max(1, total_chunks)),
        ))
        if args.save_wav:
            print("Saved wav: %s" % args.save_wav)
        if max_peak <= 0:
            print("Result: microphone opened, but no audio samples were captured. Check device routing.")
            return 2
        if args.no_asr:
            if max_rms < args.threshold:
                print(
                    "Result: audio was captured, but the RMS stayed below the advisory threshold. "
                    "The microphone is working, but gain may be low."
                )
            else:
                print("Result: microphone is receiving audio.")
            return 0
        if max_rms < args.threshold:
            print(
                "Result: microphone opened, but the RMS stayed below the advisory threshold. "
                "ASR windows may still be quiet."
            )
            return 0
        print("Result: microphone is receiving audio above threshold.")
        return 0
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except OSError as exc:
        print("ERROR: failed to start arecord: %s" % exc)
        return 1
    finally:
        if proc is not None:
            terminate_process(proc)
        if wav_file is not None:
            wav_file.close()


if __name__ == "__main__":
    sys.exit(main())
