# WPB Task 1 Owner Search Real Robot

This package migrates the simulation owner-search workflow to the real WPB Home robot through owner action recognition.

Current scope:

```text
real hardware bringup
  -> map localization and move_base navigation
  -> YOLO-World person detection on GPU
  -> InsightFace owner verification on CPU
  -> rotate scan at living_room
  -> center verified owner in the Kinect camera
  -> sample the robot camera for 7 seconds
  -> speak the owner's action
  -> if the owner is waving, approach with Kinect point cloud and ask for help
  -> if the owner is lying, split floor fall vs sofa/bed/chair lying by point-cloud height
  -> approach fallen/lying/sitting owners by odometry; run the arm assist only for falls
  -> after normal sitting/elevated lying approach, listen for and report an electrical-switch command
```

Not included yet:

```text
simulation action hint
Gazebo-specific topics
```

## Required Real Robot Inputs

The launch file starts the same core drivers used by the WPB Home examples:

- `/dev/ftdi` through `wpb_home_bringup/wp_home_core` for `/odom` and `/cmd_vel`
- `/dev/rplidar` through `rplidar_ros`, filtered to `/scan`
- `kinect2_bridge` for `/kinect2/qhd/image_color_rect`
- `jie_ware/lidar_loc` localization, `move_base`, and `wpbh_local_planner`
- Offline voice bridge with PiperTTS on `/voice/say` and `sound_play` playback; the switch-command step runs `tools/local_switch_command_test.py` as a standalone subprocess by default
- `yoloworld_perception` on `cuda:0`, with YOLO bounding-box debug image on `/perception/yoloworld/debug_image`

The task node waits for `/kinect2/qhd/image_color_rect`, `/scan`, and `/odom` before it starts moving. This is intentional for real robot safety.

Speech output is routed through `offline_voice_bridge`, not the xfyun stack. The task publishes prompt text to `/voice/say`; `offline_tts_node.py` uses PiperTTS to generate a wav and `sound_play` plays it through the audio device on the machine running the launch file. Background `offline_asr_node.py` is disabled by default in this task launch so it does not occupy the microphone; after the task says `请指示。`, the switch-command step starts `python3 -u tools/local_switch_command_test.py` as a standalone subprocess so the same microphone, ASR, LLM, TTS, and `aplay` logs are visible in the task terminal. You can still pass `start_asr:=true` when you specifically want to debug the shared `/voice/asr_text` topic.

## Files To Prepare

Put or update the upright owner reference photos here:

```bash
/home/ubuntu20/catkin_ws/src/wpb_task1_owner_search/data/owner/
```

The owner face verifier accepts either one image file or a directory of images. For better recognition from side views and downward/upward angles, place several single-person upright photos in this directory, for example `owner_front.jpg`, `owner_left.jpg`, `owner_right.jpg`, and `owner_down.jpg`. Do not use this directory for lying-owner calibration photos; keep those outside the package for validation so they do not broaden the runtime reference set. Avoid group photos or photos where the owner's face is very small; the loader uses the largest detected face in each reference photo.

During verification the node chooses the face crop from the YOLO person-box shape. If the box is taller than wide, it treats the person as upright and checks only the upper crop without rotation. If the box is wider than tall, it treats the person as lying and checks left/right side crops, because the face is often at one horizontal end of the body box.

For lying candidates, the verifier tries rotated versions of each candidate face crop, so sideways faces can still be matched against the upright owner references. Reference-photo rotation is disabled by default because it can create unstable high-similarity matches for sideways non-owners. If the first raw pass is still below threshold, the verifier uses CLAHE contrast normalization and 1.5x upsampling as a fallback for the lying side crops.

Prepare a real robot map and waypoint file:

```bash
/home/ubuntu20/catkin_ws/src/wpb_home/wpb_home_tutorials/maps/map.yaml
/home/ubuntu20/catkin_ws/src/wpb_home/wpb_home_tutorials/maps/map.pgm
/home/ubuntu20/waypoints.xml
```

Keep the map files in the real WPB Home tutorials map directory. Do not reuse the simulation package map directory.

If `map.yaml` and `map.pgm` are in the same directory, keep the YAML image entry relative:

```yaml
image: map.pgm
```

The waypoint file must contain a waypoint named:

```text
living_room
```

If your map is temporarily elsewhere, pass it through the `map:=...` launch argument.

## Run

Recommended for repeated real-robot tests: start the robot/camera/YOLO stack once, then rerun only the task node.

Terminal 1, after robot PC reboot or after fully stopping the stack:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpb_task1_owner_search task1_owner_search_bringup.launch
```

Wait until the camera and YOLO topics are alive:

```bash
rostopic hz /kinect2/qhd/image_color_rect
rostopic hz /kinect2/qhd/points
rostopic hz /perception/yoloworld/debug_image
```

Terminal 2, rerun this command for each attempt:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpb_task1_owner_search task1_owner_search_task_only.launch
```

The recommended two-terminal flow starts TTS/sound playback in
`task1_owner_search_bringup.launch`; background ASR is disabled by default
because the task runs the standalone local switch-command script after saying
`请指示。`. If only `task_only` is running, start its voice chain explicitly:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_task_only.launch start_voice:=true
```

If you explicitly want to debug the shared `/voice/asr_text` topic, add
`start_asr:=true`. The task's switch-command path does not require it in the
default `local_script` mode.

The launch defaults the ASR microphone to the ALSA `default` capture device,
which matches the `wpb_home` voice stack. Check the actual device on the robot
with `arecord -l`; if the default route is wrong, pass for example:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch \
  asr_capture_device:=plughw:CARD=YourCard,DEV=0
```

If the robot is already at the living room and you only want to test owner search and action recognition:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_task_only.launch navigate_enabled:=false
```

Single-command full launch is still available, but avoid using it repeatedly in a tight loop because it restarts Kinect2, point-cloud nodelets, YOLO, voice, RViz, and the task every time:

```bash
cd ~/catkin_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch
```

By default this opens the YOLO person-box debug window and does not show a separate raw Kinect window. `Owner YOLO-Person Box Real` shows the YOLO-World person boxes on the camera image. If you are running headless over SSH, no OpenCV window will appear unless `DISPLAY` is available; in that case view the topic in RViz/ImageView instead:

```bash
rosrun image_view image_view image:=/perception/yoloworld/debug_image
```

To disable the viewer window while keeping YOLO detections running:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch show_yolo_viewer:=false
```

If the robot base, lidar, Kinect, localization, and move_base are already running:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch start_robot:=false
```

The real launch defaults to `jie_ware/lidar_loc`, which publishes `map -> odom` from the map and `/scan`. To compare with the original AMCL path, run:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch use_jie_lidar_loc:=false
```

If the robot is already at the living room and you only want to test owner search:

```bash
roslaunch wpb_task1_owner_search task1_owner_search_real.launch start_robot:=false navigate_enabled:=false
```

For hardware debug only, if you want to test scanning without owner face verification, edit `config/task1_owner_search_real.yaml`:

```yaml
face_verify_required: false
allow_unverified_owner: true
```

Do not use that setting in the final competition flow because the robot may accept a guest as the owner.

## Owner Action Recognition

After InsightFace confirms the owner, the robot tries to center the owner in the Kinect image. If centering is unstable or times out, it silently skips centering, says `识别中。`, samples `/kinect2/qhd/image_color_rect` for 7 seconds, and announces the detected action.

The action recognizer follows the lightweight YOLO-pose sampling logic used by `wpr_simulation/scripts/action_camera_piper.py`, but it reads ROS camera frames from the real robot instead of opening a local USB camera.

Pose model path:

```bash
/home/ubuntu20/catkin_ws/src/wpr_simulation/models/vision/yolo11n-pose.pt
```

Current supported action announcements:

- owner is sitting
- owner is lying down on sofa/bed/chair
- owner may have suddenly fallen
- owner is already lying on the floor after a fall
- owner is waving
- action is uncertain

Fall detection is treated as a transition, not just a final posture. The node compares the first and last thirds of the 7-second pose window and reports a fall when the owner starts mostly upright/non-lying and ends mostly lying, with supporting motion evidence such as full-frame body center drop, torso rotation, wider body box, or reduced body-box height. This avoids classifying a one-time fall as merely `lying` when the final frames are already horizontal.

Sitting detection uses YOLO-pose keypoints plus a small evidence score instead of relying on only one perfect full-body pose. A frame can support `sitting` through bent knees, knees close to hips, one-sided knee/hip evidence, a roughly horizontal thigh, ankles folded closer to the hips, or a compact seated body box, while still requiring the torso to remain reasonably vertical and not lying-like. The final `sitting` verdict is still based on repeated evidence across the 7-second sampling window, with relaxed recovery thresholds in `config/task1_owner_search_real.yaml` for frames where one leg keypoint is missing.

When the detected action is `waving`, the real robot does not use simulation-only model-state hints or a Gazebo 3D goal. It samples `/kinect2/qhd/points` inside the verified owner's Kinect 2D person box, estimates the owner's 3D position relative to the robot, converts that relative offset from `base_footprint` into a `map` goal with TF, then sends a `move_base` goal to reach the configured standoff distance. The default waving standoff is 0.45 m with 0.05 m finish tolerance, keeping the final target within 0.50 m before asking `请问您需要什么帮助？`.

The waving approach defaults to a 25-second owner-position sampling window before handing the final approach to `move_base`. `approach_navigation_enabled: true` is the normal path; `approach_direct_fallback_enabled: false` prevents the robot from reverting to blind forward motion if `move_base` cannot plan the near-owner approach. If it cannot finish, the node logs the concrete reason and does not ask the help prompt from a far position.

When the detected action is `falling`, the robot announces `识别到主人摔倒。`, snapshots the owner's 3D position, approaches by odometry, then advances a short extra distance, and finally runs the arm assist motion. If the final pose is only classified as static `lying`, the robot first samples Kinect point-cloud surface height inside the owner box: low surfaces are treated as `lying_ground` and handled the same as a fall, while elevated surfaces are treated as sofa/bed/chair lying and announced as `识别到主人躺下。`.

Fall, non-fall lying, and sitting states use the same snapshot approach mode: the robot records a single relative 3D target and, by default, transforms the standoff point into the `map` frame before sending it to `move_base` instead of manually driving the measured distance by wheel odometry. The extra-close nudge also uses a transformed `map` goal first, so it participates in obstacle avoidance; direct `/cmd_vel` movement is only used if `approach_navigation_enabled` is disabled or `approach_direct_fallback_enabled` is explicitly enabled. `/scan` remains active as a forward safety guard. `fall_approach_fast_finish_tolerance` and `fall_approach_extra_close_finish_tolerance` prevent the final near-owner nudge from crawling for the last few centimeters. Only fall and floor-lying cases extend the arm on `/wpb_home/mani_ctrl`: extend with `name=['lift','gripper']`, hold briefly, then retract.

After the robot completes the approach and extra forward nudge for a normal `sitting` or elevated `lying` owner, it says `请指示。` and then runs `python3 -u tools/local_switch_command_test.py --until-result --count 1` as a standalone subprocess. In the default `local_script` mode, that script records 4-second ALSA microphone windows, writes the `/dev/shm/local_switch_record_zh.wav` recording, transcribes it with faster-whisper, classifies the transcript with the local Ollama endpoint (`qwen3.5:2b`), prints the `== 第 1 轮 ===` / `识别文本` / `LLM 输出` / `判断结果` / `TTS` / `aplay` lines directly in the task terminal, and plays the fixed response itself. If one round has no recognized text, the same preloaded ASR/LLM/TTS process immediately starts `第 2 轮`, then `第 3 轮`, until a transcript is classified. The task node parses the script's `判断结果:` line, records `on`, `off`, or `unknown`, and publishes it latched on `/electrical_switch/state`. Fall and floor-lying (`lying_ground`) paths do not enter this interaction and retain the arm-assist behavior. This package currently records and publishes the requested state; it does not actuate physical switch hardware because no switch-driver topic/service is defined here.

The default standalone script mode loops through 4-second instruction rounds until one round produces a switch judgment. Set `electrical_switch_script_until_result: false`, or switch `electrical_switch_instruction_source` back to `direct_asr` or `ros_topic`, only if you want the older one-shot/internal recognizer behavior.

## Useful Checks

```bash
rostopic hz /odom
rostopic hz /scan
rostopic hz /kinect2/qhd/points
rosrun tf tf_echo map odom
rosnode list | grep -E 'lidar_loc|amcl'
rostopic hz /kinect2/qhd/image_color_rect
rostopic echo /perception/person_detections_2d
rostopic hz /perception/yoloworld/debug_image
watch -n 1 nvidia-smi
rostopic info /voice/say
rostopic info /voice/asr_text
rostopic echo /electrical_switch/state
```

## Troubleshooting Repeated Runs

If `task1_find_owner_real` exits while waiting for hardware and the log says no messages arrived from `/kinect2/qhd/image_color_rect`, the Kinect topic may be advertised but publishing at 0 Hz. Check it before running the task:

```bash
rostopic hz /kinect2/qhd/image_color_rect
```

If the rate stays at 0 Hz, fully stop the launch, restart `kinect2_bridge`, or unplug/replug the Kinect USB cable before starting the task again.

For cleanup after repeated interrupted runs:

```bash
~/catkin_ws/src/wpb_task1_owner_search/tools/task1_cleanup_runtime.sh
```

If `nvidia-smi` cannot communicate with the NVIDIA driver, YOLO on `cuda:0` will not be reliable. Reboot the robot PC or fix the NVIDIA driver state before testing YOLO again.

If `move_base` reports `Failed to get a plan`, confirm that `/home/ubuntu20/waypoints.xml` still contains the intended `living_room` pose. A bad or accidentally re-saved waypoint near an obstacle can prevent planning even when localization and lidar are working.

Quick speech test:

```bash
rostopic pub -1 /voice/say std_msgs/String "data: '我已经识别到主人。'"
```

Raw robot microphone and Chinese ASR test:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --list-devices --seconds 12
```

The test first loads the local faster-whisper Chinese model, then captures
audio in 4-second windows. Speak near the robot after the command starts. If
the microphone is receiving audio, the printed RMS value should jump and lines
should show `VOICE`; each active window also prints `识别文本`. To test only
the microphone level without loading ASR:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --no-asr --seconds 10
```

To save the captured audio for playback:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --seconds 10 --save-wav /tmp/robot_mic_test.wav
aplay /tmp/robot_mic_test.wav
```

If `default` does not work but `arecord -l` shows another capture card, pass it
explicitly:

```bash
rosrun wpb_task1_owner_search robot_mic_test.py --device plughw:CARD=Generic_1,DEV=0 --seconds 10
```

Expected task completion log:

```text
Owner accepted at bbox=..., confidence=..., reason=face accepted
Owner centered in camera: center=...
Recognizing owner action for 7.0s with robot camera
Owner action verdict: label=..., confidence=..., samples=...
task1_find_owner_real finished: success=True
```
