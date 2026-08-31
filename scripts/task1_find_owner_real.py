#!/usr/bin/env python3
# coding: utf-8
import math
import audioop
import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import wave
import xml.etree.ElementTree as ET

import actionlib
import cv2
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from actionlib_msgs.msg import GoalStatus
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from perception_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty


def clamp(value, low, high):
    return max(low, min(high, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def signed_angle_diff(target, current):
    return math.atan2(math.sin(target - current), math.cos(target - current))


class MissingHardwareError(RuntimeError):
    """Raised when required robot sensor streams are not producing messages."""


class RealOwnerSearchBeforeAction:
    """Real WPB owner-search task with owner centering and action recognition."""

    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image = None
        self.latest_image_time = None
        self.latest_detections = []
        self.latest_detections_time = None
        self.latest_scan = None
        self.latest_scan_time = None
        self.latest_pointcloud = None
        self.latest_pointcloud_time = None
        self.latest_yaw = None
        self.latest_odom_xy = None
        self.latest_odom_time = None
        self.owner_track_center = None
        self.last_pointcloud_reason = ""
        self.last_approach_failure_reason = ""

        self.waypoint_name = rospy.get_param("~waypoint_name", "living_room")
        self.waypoint_file = os.path.expanduser(rospy.get_param("~waypoint_file", "~/waypoints.xml"))
        self.owner_image_path = os.path.expanduser(rospy.get_param("~owner_image_path", ""))

        self.image_topic = rospy.get_param("~image_topic", "/kinect2/qhd/image_color_rect")
        self.detections_topic = rospy.get_param("~detections_topic", "/perception/person_detections_2d")
        self.points_topic = rospy.get_param("~points_topic", "/kinect2/qhd/points")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.say_topic = rospy.get_param("~say_topic", "/voice/say")
        self.asr_topic = rospy.get_param("~asr_topic", "/voice/asr_text")
        self.electrical_switch_state_topic = rospy.get_param(
            "~electrical_switch_state_topic", "/electrical_switch/state"
        )
        self.pause_yolo_topic = rospy.get_param("~pause_yolo_topic", "/yoloworld/pause")

        self.require_kinect = bool(rospy.get_param("~require_kinect", True))
        self.require_lidar = bool(rospy.get_param("~require_lidar", True))
        self.require_odom = bool(rospy.get_param("~require_odom", True))
        self.hardware_check_timeout = float(rospy.get_param("~hardware_check_timeout", 90.0))

        self.clear_costmaps_before_navigation = bool(rospy.get_param("~clear_costmaps_before_navigation", True))
        self.retry_navigation_after_clear = bool(rospy.get_param("~retry_navigation_after_clear", True))
        self.clear_costmaps_service = rospy.get_param("~clear_costmaps_service", "/move_base/clear_costmaps")
        self.clear_costmaps_timeout = float(rospy.get_param("~clear_costmaps_timeout", 5.0))

        self.face_verify_enabled = bool(rospy.get_param("~face_verify_enabled", True))
        self.face_verify_required = bool(rospy.get_param("~face_verify_required", True))
        self.allow_unverified_owner = bool(rospy.get_param("~allow_unverified_owner", False))
        self.face_auto_download = bool(rospy.get_param("~face_auto_download", False))
        self.face_model_name = rospy.get_param("~face_model_name", "buffalo_sc")
        self.face_model_root = os.path.expanduser(rospy.get_param("~face_model_root", "~/.insightface"))
        self.face_ctx_id = int(rospy.get_param("~face_ctx_id", -1))
        self.face_det_size = int(rospy.get_param("~face_det_size", 480))
        self.face_det_thresh = float(rospy.get_param("~face_det_thresh", 0.35))
        self.face_accept_threshold = float(rospy.get_param("~face_accept_threshold", 0.45))
        self.face_reject_threshold = float(rospy.get_param("~face_reject_threshold", 0.25))
        self.face_fast_reject = bool(rospy.get_param("~face_fast_reject", True))
        self.face_reference_try_rotations = bool(rospy.get_param("~face_reference_try_rotations", False))
        self.face_crop_padding = float(rospy.get_param("~face_crop_padding", 0.28))
        self.face_crop_top_ratio = float(rospy.get_param("~face_crop_top_ratio", 0.68))
        self.face_crop_lying_side_ratio = float(rospy.get_param("~face_crop_lying_side_ratio", 0.50))
        self.face_crop_lying_extra_side_ratios = self.parse_float_list(
            rospy.get_param("~face_crop_lying_extra_side_ratios", [0.45, 0.65, 0.75])
        )
        self.face_crop_try_rotations = bool(rospy.get_param("~face_crop_try_rotations", True))
        self.face_candidate_enhance = bool(rospy.get_param("~face_candidate_enhance", True))
        self.face_fast_pass_enabled = bool(rospy.get_param("~face_fast_pass_enabled", True))
        self.face_fast_pass_reject_threshold = float(
            rospy.get_param("~face_fast_pass_reject_threshold", 0.30)
        )

        self.detection_min_score = float(rospy.get_param("~detection_min_score", 0.30))
        self.detection_min_area_ratio = float(rospy.get_param("~detection_min_area_ratio", 0.015))
        self.verify_top_k = max(1, int(rospy.get_param("~verify_top_k", 1)))
        self.verify_during_scan = bool(rospy.get_param("~verify_during_scan", True))
        self.verify_after_scan_top_k = max(1, int(rospy.get_param("~verify_after_scan_top_k", 1)))
        self.candidate_cooldown = float(rospy.get_param("~candidate_cooldown", 1.0))
        self.candidate_collection_interval = float(rospy.get_param("~candidate_collection_interval", 0.5))
        self.scan_candidate_pool_size = max(1, int(rospy.get_param("~scan_candidate_pool_size", 6)))

        self.navigate_enabled = bool(rospy.get_param("~navigate_enabled", True))
        self.navigate_timeout = float(rospy.get_param("~navigate_timeout", 120.0))
        self.scan_angular_speed = float(rospy.get_param("~scan_angular_speed", -0.25))
        self.scan_duration = float(rospy.get_param("~scan_duration", 13.0))
        self.scan_total_angle = float(rospy.get_param("~scan_total_angle", math.pi))
        self.scan_timeout = float(rospy.get_param("~scan_timeout", 35.0))
        self.scan_after_arrival_delay = float(rospy.get_param("~scan_after_arrival_delay", 0.2))
        self.scan_return_to_owner = bool(rospy.get_param("~scan_return_to_owner", True))
        self.return_angular_speed = abs(float(rospy.get_param("~return_angular_speed", 0.25)))

        self.center_owner_enabled = bool(rospy.get_param("~center_owner_enabled", True))
        self.center_owner_timeout = float(rospy.get_param("~center_owner_timeout", 5.0))
        self.center_owner_tolerance = float(rospy.get_param("~center_owner_tolerance", 0.06))
        self.center_owner_angular_gain = float(rospy.get_param("~center_owner_angular_gain", 0.65))
        self.center_owner_max_angular_speed = abs(float(rospy.get_param("~center_owner_max_angular_speed", 0.30)))
        self.center_owner_lost_turn_speed = abs(float(rospy.get_param("~center_owner_lost_turn_speed", 0.10)))

        self.action_recognition_enabled = bool(rospy.get_param("~action_recognition_enabled", True))
        self.action_model_path = os.path.expanduser(rospy.get_param("~action_model_path", ""))
        self.action_device = rospy.get_param("~action_device", "cuda:0")
        self.action_require_gpu = bool(rospy.get_param("~action_require_gpu", True))
        self.action_imgsz = int(rospy.get_param("~action_imgsz", 416))
        self.action_conf = float(rospy.get_param("~action_conf", 0.30))
        self.action_iou = float(rospy.get_param("~action_iou", 0.45))
        self.action_half = bool(rospy.get_param("~action_half", True))
        self.action_sample_seconds = float(rospy.get_param("~action_sample_seconds", 7.0))
        self.action_sample_rate = float(rospy.get_param("~action_sample_rate", 3.0))
        self.action_min_keypoint_conf = float(rospy.get_param("~action_min_keypoint_conf", 0.25))
        self.action_max_det = int(rospy.get_param("~action_max_det", 4))
        self.action_pause_yolo = bool(rospy.get_param("~action_pause_yolo", False))
        self.action_use_owner_roi = bool(rospy.get_param("~action_use_owner_roi", True))
        self.action_roi_padding = float(rospy.get_param("~action_roi_padding", 0.35))
        self.action_min_pose_samples = int(rospy.get_param("~action_min_pose_samples", 5))
        self.action_static_required_ratio = float(rospy.get_param("~action_static_required_ratio", 0.65))
        self.action_fall_center_drop = float(rospy.get_param("~action_fall_center_drop", 0.06))
        self.action_fall_torso_drop = float(rospy.get_param("~action_fall_torso_drop", 0.20))
        self.action_fall_aspect_gain = float(rospy.get_param("~action_fall_aspect_gain", 0.25))
        self.action_fall_lie_ratio_gain = float(rospy.get_param("~action_fall_lie_ratio_gain", 0.35))
        self.action_fall_late_lie_ratio = float(rospy.get_param("~action_fall_late_lie_ratio", 0.45))
        self.action_fall_height_shrink_ratio = float(rospy.get_param("~action_fall_height_shrink_ratio", 0.82))
        self.action_fall_early_upright_ratio = float(rospy.get_param("~action_fall_early_upright_ratio", 0.35))
        self.lying_surface_classification_enabled = bool(
            rospy.get_param("~lying_surface_classification_enabled", True)
        )
        self.lying_surface_camera_height = float(rospy.get_param("~lying_surface_camera_height", 0.85))
        self.lying_ground_max_surface_height = float(rospy.get_param("~lying_ground_max_surface_height", 0.35))
        self.lying_furniture_min_surface_height = float(rospy.get_param("~lying_furniture_min_surface_height", 0.42))
        self.lying_ground_bbox_bottom_ratio = float(rospy.get_param("~lying_ground_bbox_bottom_ratio", 0.88))
        self.lying_ground_bbox_center_ratio = float(rospy.get_param("~lying_ground_bbox_center_ratio", 0.62))
        self.action_report_standing = bool(rospy.get_param("~action_report_standing", False))
        self.action_speech_hold = float(rospy.get_param("~action_speech_hold", 3.2))

        self.approach_on_waving_enabled = bool(rospy.get_param("~approach_on_waving_enabled", True))
        self.approach_timeout = float(rospy.get_param("~approach_timeout", 25.0))
        self.approach_linear_speed = abs(float(rospy.get_param("~approach_linear_speed", 0.18)))
        self.approach_angular_gain = float(rospy.get_param("~approach_angular_gain", 0.45))
        self.approach_max_angular_speed = abs(float(rospy.get_param("~approach_max_angular_speed", 0.25)))
        self.approach_stop_distance = float(rospy.get_param("~approach_stop_distance", 0.70))
        self.approach_standoff_distance = float(rospy.get_param("~approach_standoff_distance", self.approach_stop_distance))
        self.approach_distance_tolerance = float(rospy.get_param("~approach_distance_tolerance", 0.08))
        self.approach_bearing_tolerance = float(rospy.get_param("~approach_bearing_tolerance", 0.06))
        self.approach_forward_bearing_limit = float(rospy.get_param("~approach_forward_bearing_limit", 0.45))
        self.approach_arrival_stable_cycles = max(1, int(rospy.get_param("~approach_arrival_stable_cycles", 1)))
        self.approach_missing_data_grace = float(rospy.get_param("~approach_missing_data_grace", 0.7))
        self.approach_missing_linear_scale = float(rospy.get_param("~approach_missing_linear_scale", 0.35))
        self.approach_command_smoothing = float(rospy.get_param("~approach_command_smoothing", 0.45))
        self.approach_center_tolerance = float(rospy.get_param("~approach_center_tolerance", 0.12))
        self.approach_target_height_ratio = float(rospy.get_param("~approach_target_height_ratio", 0.55))
        self.approach_lost_turn_speed = abs(float(rospy.get_param("~approach_lost_turn_speed", 0.10)))
        self.approach_front_scan_degrees = float(rospy.get_param("~approach_front_scan_degrees", 12.0))
        self.approach_scan_max_age = float(rospy.get_param("~approach_scan_max_age", 0.8))
        self.approach_require_lidar = bool(rospy.get_param("~approach_require_lidar", False))
        self.approach_lidar_stop_distance = float(rospy.get_param("~approach_lidar_stop_distance", 0.55))
        self.approach_lidar_margin = float(rospy.get_param("~approach_lidar_margin", 0.08))
        self.approach_pointcloud_mode = str(rospy.get_param("~approach_pointcloud_mode", "auto")).lower()
        self.approach_pointcloud_max_age = float(rospy.get_param("~approach_pointcloud_max_age", 1.0))
        self.approach_pointcloud_min_samples = max(5, int(rospy.get_param("~approach_pointcloud_min_samples", 30)))
        self.approach_pointcloud_stride = max(1, int(rospy.get_param("~approach_pointcloud_stride", 8)))
        self.approach_pointcloud_roi_x_margin = float(rospy.get_param("~approach_pointcloud_roi_x_margin", 0.25))
        self.approach_pointcloud_roi_y_min_ratio = float(rospy.get_param("~approach_pointcloud_roi_y_min_ratio", 0.15))
        self.approach_pointcloud_roi_y_max_ratio = float(rospy.get_param("~approach_pointcloud_roi_y_max_ratio", 0.85))
        self.approach_pointcloud_depth_percentile = float(rospy.get_param("~approach_pointcloud_depth_percentile", 35.0))
        self.approach_pointcloud_surface_band = float(rospy.get_param("~approach_pointcloud_surface_band", 0.30))
        self.approach_min_depth = float(rospy.get_param("~approach_min_depth", 0.35))
        self.approach_max_depth = float(rospy.get_param("~approach_max_depth", 5.0))
        self.approach_linear_gain = float(rospy.get_param("~approach_linear_gain", 0.25))
        self.approach_min_linear_speed = abs(float(rospy.get_param("~approach_min_linear_speed", 0.06)))
        self.approach_help_prompt = rospy.get_param("~approach_help_prompt", "请问您需要什么帮助？")

        self.fall_approach_enabled = bool(rospy.get_param("~fall_approach_enabled", True))
        self.fall_approach_action_labels = self.parse_string_list(
            rospy.get_param("~fall_approach_action_labels", ["falling", "lying_ground", "lying", "sitting"])
        )
        self.fall_approach_position_sample_seconds = float(rospy.get_param("~fall_approach_position_sample_seconds", 0.9))
        self.fall_approach_min_position_samples = max(
            1,
            int(rospy.get_param("~fall_approach_min_position_samples", 2)),
        )
        self.fall_approach_standoff_distance = float(rospy.get_param("~fall_approach_standoff_distance", 0.75))
        self.fall_approach_distance_tolerance = float(rospy.get_param("~fall_approach_distance_tolerance", 0.08))
        self.fall_approach_linear_speed = abs(float(rospy.get_param("~fall_approach_linear_speed", 0.14)))
        self.fall_approach_min_linear_speed = abs(float(rospy.get_param("~fall_approach_min_linear_speed", 0.05)))
        self.fall_approach_linear_gain = float(rospy.get_param("~fall_approach_linear_gain", self.approach_linear_gain))
        self.fall_approach_turn_timeout = float(rospy.get_param("~fall_approach_turn_timeout", 8.0))
        self.fall_approach_drive_timeout = float(rospy.get_param("~fall_approach_drive_timeout", 18.0))
        self.fall_approach_max_travel_distance = float(rospy.get_param("~fall_approach_max_travel_distance", 1.50))
        self.fall_approach_lidar_stop_distance = float(
            rospy.get_param("~fall_approach_lidar_stop_distance", self.approach_lidar_stop_distance)
        )
        self.fall_approach_lidar_margin = float(rospy.get_param("~fall_approach_lidar_margin", self.approach_lidar_margin))
        self.fall_approach_extra_close_enabled = bool(rospy.get_param("~fall_approach_extra_close_enabled", True))
        self.fall_approach_extra_close_distance = float(rospy.get_param("~fall_approach_extra_close_distance", 0.18))
        self.fall_approach_extra_close_speed = abs(float(rospy.get_param("~fall_approach_extra_close_speed", 0.08)))
        self.fall_approach_extra_close_timeout = float(rospy.get_param("~fall_approach_extra_close_timeout", 6.0))

        self.fall_assist_arm_enabled = bool(rospy.get_param("~fall_assist_arm_enabled", True))
        self.fall_assist_arm_action_labels = self.parse_string_list(
            rospy.get_param("~fall_assist_arm_action_labels", ["falling", "lying_ground"])
        )
        self.mani_ctrl_topic = rospy.get_param("~mani_ctrl_topic", "/wpb_home/mani_ctrl")
        self.fall_assist_arm_extend_lift = float(rospy.get_param("~fall_assist_arm_extend_lift", 0.50))
        self.fall_assist_arm_extend_gripper = float(rospy.get_param("~fall_assist_arm_extend_gripper", 0.12))
        self.fall_assist_arm_retract_lift = float(rospy.get_param("~fall_assist_arm_retract_lift", 0.0))
        self.fall_assist_arm_retract_gripper = float(rospy.get_param("~fall_assist_arm_retract_gripper", 0.12))
        self.fall_assist_arm_lift_velocity = float(rospy.get_param("~fall_assist_arm_lift_velocity", 0.5))
        self.fall_assist_arm_gripper_velocity = float(rospy.get_param("~fall_assist_arm_gripper_velocity", 5.0))
        self.fall_assist_arm_extend_wait = float(rospy.get_param("~fall_assist_arm_extend_wait", 3.0))
        self.fall_assist_arm_hold_seconds = float(rospy.get_param("~fall_assist_arm_hold_seconds", 4.0))
        self.fall_assist_arm_retract_wait = float(rospy.get_param("~fall_assist_arm_retract_wait", 3.0))
        self.fall_assist_arm_command_rate = float(rospy.get_param("~fall_assist_arm_command_rate", 5.0))

        self.electrical_switch_instruction_enabled = bool(
            rospy.get_param("~electrical_switch_instruction_enabled", True)
        )
        self.electrical_switch_instruction_source = str(
            rospy.get_param("~electrical_switch_instruction_source", "direct_asr")
        ).strip().lower()
        self.electrical_switch_instruction_timeout = float(
            rospy.get_param("~electrical_switch_instruction_timeout", 0.0)
        )
        self.electrical_switch_instruction_window_seconds = float(
            rospy.get_param("~electrical_switch_instruction_window_seconds", 5.0)
        )
        self.electrical_switch_instruction_asr_settle_seconds = float(
            rospy.get_param("~electrical_switch_instruction_asr_settle_seconds", 1.0)
        )
        self.electrical_switch_instruction_max_empty_windows = int(
            rospy.get_param("~electrical_switch_instruction_max_empty_windows", 0)
        )
        self.electrical_switch_prompt = rospy.get_param("~electrical_switch_prompt", "请指示。")
        self.electrical_switch_prompt_hold = float(rospy.get_param("~electrical_switch_prompt_hold", 1.5))
        script_dir = os.path.dirname(os.path.abspath(__file__))
        catkin_src_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        default_switch_asr_model = os.path.join(
            catkin_src_dir,
            "offline_voice_bridge",
            "models",
            "whisper",
            "faster-whisper-small",
        )
        self.electrical_switch_asr_capture_device = rospy.get_param(
            "~electrical_switch_asr_capture_device", rospy.get_param("/asr/capture_device", "default")
        )
        self.electrical_switch_asr_sample_rate = int(rospy.get_param("~electrical_switch_asr_sample_rate", 16000))
        self.electrical_switch_asr_channels = int(rospy.get_param("~electrical_switch_asr_channels", 1))
        self.electrical_switch_asr_model_path = os.path.expanduser(
            rospy.get_param("~electrical_switch_asr_model_path", default_switch_asr_model)
        )
        self.electrical_switch_asr_hf_endpoint = rospy.get_param(
            "~electrical_switch_asr_hf_endpoint", "https://huggingface.co"
        )
        self.electrical_switch_asr_device = rospy.get_param("~electrical_switch_asr_device", "cpu")
        self.electrical_switch_asr_compute_type = rospy.get_param("~electrical_switch_asr_compute_type", "int8")
        self.electrical_switch_asr_language = rospy.get_param("~electrical_switch_asr_language", "zh")
        self.electrical_switch_asr_beam_size = int(rospy.get_param("~electrical_switch_asr_beam_size", 1))
        self.electrical_switch_asr_vad_filter = bool(rospy.get_param("~electrical_switch_asr_vad_filter", False))
        self.electrical_switch_asr_no_speech_threshold = float(
            rospy.get_param("~electrical_switch_asr_no_speech_threshold", 0.6)
        )
        self.electrical_switch_asr_keep_wav = bool(rospy.get_param("~electrical_switch_asr_keep_wav", False))
        self.electrical_switch_asr_wav_dir = rospy.get_param("~electrical_switch_asr_wav_dir", "/dev/shm")
        self.electrical_switch_ollama_url = rospy.get_param(
            "~electrical_switch_ollama_url", "http://127.0.0.1:11434/api/chat"
        )
        self.electrical_switch_ollama_model = rospy.get_param(
            "~electrical_switch_ollama_model", "qwen3.5:2b"
        )
        self.electrical_switch_ollama_timeout = float(
            rospy.get_param("~electrical_switch_ollama_timeout", 30.0)
        )
        self.electrical_switch_ollama_keep_alive = rospy.get_param(
            "~electrical_switch_ollama_keep_alive", "10m"
        )
        self.electrical_switch_ollama_max_tokens = max(
            8, int(rospy.get_param("~electrical_switch_ollama_max_tokens", 20))
        )
        self.electrical_switch_reply_on = rospy.get_param(
            "~electrical_switch_reply_on", "好的，已开启电气开关。"
        )
        self.electrical_switch_reply_off = rospy.get_param(
            "~electrical_switch_reply_off", "好的，已关闭电气开关。"
        )
        self.electrical_switch_reply_unknown = rospy.get_param(
            "~electrical_switch_reply_unknown", "抱歉，我没有听出开关指令。"
        )
        self.latest_asr_text = ""
        self.latest_asr_time = None
        self.asr_sequence = 0
        self.asr_history = []
        self.electrical_switch_asr_model = None
        self.electrical_switch_asr_ready = False
        self.electrical_switch_state = "unknown"

        self.speak_on_start = bool(rospy.get_param("~speak_on_start", True))
        self.speak_on_arrival = bool(rospy.get_param("~speak_on_arrival", False))
        self.speak_on_owner_found = bool(rospy.get_param("~speak_on_owner_found", True))
        self.speak_on_finish = bool(rospy.get_param("~speak_on_finish", True))
        self.say_wait_for_subscribers = bool(rospy.get_param("~say_wait_for_subscribers", True))
        self.say_wait_timeout = float(rospy.get_param("~say_wait_timeout", 20.0))
        self.say_repeat_count = max(1, int(rospy.get_param("~say_repeat_count", 1)))
        self.say_repeat_interval = float(rospy.get_param("~say_repeat_interval", 0.25))
        self.say_after_publish_delay = float(rospy.get_param("~say_after_publish_delay", 1.0))

        self.owner_reference_images = []
        self.face_app = None
        self.owner_face_embedding = None
        self.owner_face_embeddings = []
        self.face_ready = False
        self.action_pose_model = None
        self.action_ready = False

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.say_pub = rospy.Publisher(self.say_topic, String, queue_size=5)
        self.electrical_switch_state_pub = rospy.Publisher(
            self.electrical_switch_state_topic, String, queue_size=1, latch=True
        )
        self.pause_yolo_pub = rospy.Publisher(self.pause_yolo_topic, Bool, queue_size=1, latch=True)
        self.mani_ctrl_pub = rospy.Publisher(self.mani_ctrl_topic, JointState, queue_size=5)

        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.det_sub = rospy.Subscriber(self.detections_topic, Detection2DArray, self.detections_callback, queue_size=1)
        self.points_sub = rospy.Subscriber(self.points_topic, PointCloud2, self.pointcloud_callback, queue_size=1)
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.asr_sub = rospy.Subscriber(self.asr_topic, String, self.asr_callback, queue_size=10)

        self.publish_electrical_switch_state()

        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        if self.face_verify_enabled:
            self.owner_reference_images = self.load_owner_images()
            self.init_face_recognizer()
        if self.face_verify_required and not self.face_ready and not self.allow_unverified_owner:
            raise RuntimeError(
                "face verification is required but not ready; check owner_image_path and InsightFace model cache"
            )
        self.init_action_recognizer()

    def load_owner_images(self):
        if not self.owner_image_path:
            raise RuntimeError("owner_image_path is empty")
        if not os.path.exists(self.owner_image_path):
            raise RuntimeError("owner photo path not found: %s" % self.owner_image_path)

        image_paths = []
        if os.path.isdir(self.owner_image_path):
            extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            for name in sorted(os.listdir(self.owner_image_path)):
                path = os.path.join(self.owner_image_path, name)
                if os.path.isfile(path) and name.lower().endswith(extensions):
                    image_paths.append(path)
            if not image_paths:
                raise RuntimeError("no owner reference photos found under: %s" % self.owner_image_path)
        else:
            image_paths.append(self.owner_image_path)

        images = []
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                rospy.logwarn("Skipping unreadable owner reference photo: %s", path)
                continue
            images.append((path, image))

        if not images:
            raise RuntimeError("failed to read any owner reference photos from: %s" % self.owner_image_path)
        return images

    @staticmethod
    def normalize_embedding(embedding):
        vector = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-6:
            return None
        return vector / norm

    @staticmethod
    def parse_float_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]

        parsed = []
        for item in raw_values:
            try:
                parsed.append(float(item))
            except (TypeError, ValueError):
                continue
        return parsed

    @staticmethod
    def parse_string_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            raw_values = value.replace(";", ",").split(",")
        elif isinstance(value, (list, tuple)):
            raw_values = value
        else:
            raw_values = [value]

        parsed = []
        for item in raw_values:
            text = str(item).strip().lower()
            if text:
                parsed.append(text)
        return parsed

    @staticmethod
    def select_largest_face(faces):
        if not faces:
            return None

        def face_priority(face):
            bbox = getattr(face, "bbox", None)
            if bbox is None or len(bbox) < 4:
                return 0.0
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            det_score = float(getattr(face, "det_score", 1.0) or 1.0)
            return width * height * det_score

        return max(faces, key=face_priority)

    def face_reference_orientations(self, path, image):
        yield path, image
        if not self.face_reference_try_rotations:
            return
        yield path + ":rot90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        yield path + ":rot270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yield path + ":rot180", cv2.rotate(image, cv2.ROTATE_180)

    def init_face_recognizer(self):
        model_dir = os.path.join(self.face_model_root, "models", self.face_model_name)
        cached_models = []
        if os.path.isdir(model_dir):
            cached_models = [name for name in os.listdir(model_dir) if name.endswith(".onnx")]
        if not cached_models and not self.face_auto_download:
            rospy.logwarn(
                "InsightFace model %s is not cached under %s. Run the install tool first or set face_auto_download=true.",
                self.face_model_name,
                model_dir,
            )
            return

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            rospy.logwarn("InsightFace is not available: %s", exc)
            return

        providers = ["CPUExecutionProvider"] if self.face_ctx_id < 0 else None
        try:
            self.face_app = FaceAnalysis(
                name=self.face_model_name,
                root=self.face_model_root,
                providers=providers,
            )
            self.face_app.prepare(
                ctx_id=self.face_ctx_id,
                det_thresh=self.face_det_thresh,
                det_size=(self.face_det_size, self.face_det_size),
            )
            owner_embeddings = []
            for path, image in self.owner_reference_images:
                loaded_for_photo = 0
                for oriented_path, oriented_image in self.face_reference_orientations(path, image):
                    faces = self.face_app.get(oriented_image)
                    owner_face = self.select_largest_face(faces)
                    if owner_face is None:
                        continue
                    embedding = self.normalize_embedding(owner_face.embedding)
                    if embedding is None:
                        continue
                    owner_embeddings.append({"path": oriented_path, "embedding": embedding})
                    loaded_for_photo += 1
                if loaded_for_photo == 0:
                    rospy.logwarn("No face found in owner reference photo, skipping: %s", path)
            if not owner_embeddings:
                rospy.logwarn("No usable owner face embeddings loaded from: %s", self.owner_image_path)
                return
            mean_embedding = self.normalize_embedding(
                np.mean(np.asarray([item["embedding"] for item in owner_embeddings], dtype=np.float32), axis=0)
            )
            self.owner_face_embedding = mean_embedding
            self.owner_face_embeddings = owner_embeddings
            self.face_ready = True
            rospy.loginfo(
                "Face verification ready: model=%s ctx_id=%d det_size=%d det_thresh=%.2f threshold=%.2f references=%d photos=%d ref_rotations=%s",
                self.face_model_name,
                self.face_ctx_id,
                self.face_det_size,
                self.face_det_thresh,
                self.face_accept_threshold,
                len(self.owner_face_embeddings),
                len(self.owner_reference_images),
                str(self.face_reference_try_rotations),
            )
        except Exception as exc:
            self.face_app = None
            self.owner_face_embedding = None
            self.owner_face_embeddings = []
            self.face_ready = False
            rospy.logwarn("Failed to initialize face verification: %s", exc)

    def init_action_recognizer(self):
        if not self.action_recognition_enabled:
            rospy.loginfo("Owner action recognition disabled")
            return
        if not self.action_model_path:
            rospy.logwarn("Owner action recognition disabled: action_model_path is empty")
            return
        if not os.path.exists(self.action_model_path):
            rospy.logwarn("Owner action pose model not found: %s", self.action_model_path)
            return

        if self.action_require_gpu and str(self.action_device).startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    rospy.logwarn("Owner action recognition requires GPU, but PyTorch CUDA is unavailable")
                    return
            except Exception as exc:
                rospy.logwarn("Owner action recognition cannot check CUDA availability: %s", exc)
                return

        try:
            from ultralytics import YOLO
            self.action_pose_model = YOLO(self.action_model_path)
            self.action_ready = True
            rospy.loginfo(
                "Owner action recognition ready: model=%s device=%s seconds=%.1f",
                self.action_model_path,
                self.action_device,
                self.action_sample_seconds,
            )
        except Exception as exc:
            self.action_pose_model = None
            self.action_ready = False
            rospy.logwarn("Failed to initialize owner action recognition: %s", exc)

    def snapshot_owner_action_image(self):
        with self.lock:
            if self.latest_image is None:
                return None, None
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        full_meta = {
            "frame_width": width,
            "frame_height": height,
            "crop_box": None,
            "debug_image": image,
            "pose_target_center_norm": self.owner_track_center if self.owner_track_center is not None else 0.5,
        }

        if not self.action_use_owner_roi or not detections:
            return image, full_meta

        best = None
        best_score = -1e9
        target_center = self.owner_track_center if self.owner_track_center is not None else 0.5
        for det in detections:
            class_name = getattr(det, "class_name", "")
            if class_name and class_name != "person":
                continue
            if float(det.score) < self.detection_min_score:
                continue
            xmin = clamp(int(det.xmin), 0, width - 1)
            ymin = clamp(int(det.ymin), 0, height - 1)
            xmax = clamp(int(det.xmax), 0, width - 1)
            ymax = clamp(int(det.ymax), 0, height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue

            center_norm = float(det.center_x) / max(1.0, float(width))
            area_ratio = float((xmax - xmin) * (ymax - ymin)) / max(1.0, float(width * height))
            score = float(det.score) + area_ratio * 2.0 - abs(center_norm - target_center) * 2.0
            if score > best_score:
                best_score = score
                det_center_y = float(getattr(det, "center_y", (float(ymin) + float(ymax)) * 0.5))
                best = (xmin, ymin, xmax, ymax, center_norm, float(det.center_x), det_center_y)

        if best is None:
            return image, full_meta

        xmin, ymin, xmax, ymax, center_norm, det_center_x, det_center_y = best
        box_w = xmax - xmin
        box_h = ymax - ymin
        pad_x = int(box_w * clamp(self.action_roi_padding, 0.0, 1.0))
        pad_y = int(box_h * clamp(self.action_roi_padding, 0.0, 1.0))
        cx1 = clamp(xmin - pad_x, 0, width - 1)
        cy1 = clamp(ymin - pad_y, 0, height - 1)
        cx2 = clamp(xmax + pad_x, 0, width - 1)
        cy2 = clamp(ymax + pad_y, 0, height - 1)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return image, full_meta

        self.owner_track_center = center_norm
        crop_width = max(1.0, float(cx2 - cx1))
        pose_target_center_norm = clamp((det_center_x - float(cx1)) / crop_width, 0.0, 1.0)
        return crop, {
            "frame_width": width,
            "frame_height": height,
            "crop_box": (cx1, cy1, cx2, cy2),
            "debug_image": image,
            "det_bbox": (xmin, ymin, xmax, ymax),
            "det_center_y_norm": det_center_y / max(1.0, float(height)),
            "det_aspect": float(box_w) / max(1.0, float(box_h)),
            "det_height_norm": float(box_h) / max(1.0, float(height)),
            "pose_target_center_norm": pose_target_center_norm,
        }

    def select_owner_pose(self, result, image_width, target_center=None):
        if result is None or result.keypoints is None or result.boxes is None:
            return None
        try:
            keypoints_xy = result.keypoints.xy.cpu().numpy()
            keypoints_conf = result.keypoints.conf.cpu().numpy()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_conf = result.boxes.conf.cpu().numpy()
        except Exception as exc:
            rospy.logwarn("Failed to read pose result tensors: %s", exc)
            return None

        count = min(len(keypoints_xy), len(boxes_xyxy))
        if count <= 0:
            return None

        best = None
        best_score = -1e9
        if target_center is None:
            target_center = self.owner_track_center
        for index in range(count):
            kxy = keypoints_xy[index]
            kconf = keypoints_conf[index]
            valid_count = int(np.sum(kconf >= self.action_min_keypoint_conf))
            if valid_count < 5:
                continue

            bbox = boxes_xyxy[index]
            center_norm = ((float(bbox[0]) + float(bbox[2])) * 0.5) / max(1.0, float(image_width))
            center_penalty = abs(center_norm - target_center) if target_center is not None else abs(center_norm - 0.5)
            score = float(boxes_conf[index]) + valid_count * 0.04 - center_penalty * 1.5
            if score > best_score:
                best_score = score
                best = {
                    "keypoints_xy": kxy,
                    "keypoints_conf": kconf,
                    "bbox": bbox,
                    "box_conf": float(boxes_conf[index]),
                    "center_norm": center_norm,
                    "valid_count": valid_count,
                }
        return best

    @staticmethod
    def angle_at(point_a, point_b, point_c):
        vec_a = np.asarray(point_a, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
        vec_c = np.asarray(point_c, dtype=np.float32) - np.asarray(point_b, dtype=np.float32)
        denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_c))
        if denom <= 1e-6:
            return None
        cosine = clamp(float(np.dot(vec_a, vec_c) / denom), -1.0, 1.0)
        return math.degrees(math.acos(cosine))

    @staticmethod
    def median_value(values):
        cleaned = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        if not cleaned:
            return None
        return float(np.median(np.asarray(cleaned, dtype=np.float32)))

    def build_pose_feature(self, pose, image_shape, source_meta=None):
        height, width = image_shape[:2]
        kxy = pose["keypoints_xy"]
        kconf = pose["keypoints_conf"]
        bbox = pose["bbox"]
        box_w = max(1.0, float(bbox[2]) - float(bbox[0]))
        box_h = max(1.0, float(bbox[3]) - float(bbox[1]))
        aspect = box_w / box_h
        center_y_norm = ((float(bbox[1]) + float(bbox[3])) * 0.5) / max(1.0, float(height))
        full_center_y_norm = center_y_norm
        full_box_height_norm = box_h / max(1.0, float(height))
        full_aspect = aspect
        if source_meta:
            frame_height = float(source_meta.get("frame_height") or height)
            crop_box = source_meta.get("crop_box")
            if crop_box is not None:
                crop_x1, crop_y1, _crop_x2, _crop_y2 = [float(value) for value in crop_box]
                full_x1 = crop_x1 + float(bbox[0])
                full_y1 = crop_y1 + float(bbox[1])
                full_x2 = crop_x1 + float(bbox[2])
                full_y2 = crop_y1 + float(bbox[3])
                full_box_w = max(1.0, full_x2 - full_x1)
                full_box_h = max(1.0, full_y2 - full_y1)
                full_center_y_norm = ((full_y1 + full_y2) * 0.5) / max(1.0, frame_height)
                full_box_height_norm = full_box_h / max(1.0, frame_height)
                full_aspect = full_box_w / full_box_h
            det_center_y_norm = source_meta.get("det_center_y_norm")
            if det_center_y_norm is not None:
                full_center_y_norm = float(det_center_y_norm)
            det_aspect = source_meta.get("det_aspect")
            if det_aspect is not None and math.isfinite(float(det_aspect)):
                full_aspect = max(full_aspect, float(det_aspect))
            det_height_norm = source_meta.get("det_height_norm")
            if det_height_norm is not None and math.isfinite(float(det_height_norm)):
                full_box_height_norm = float(det_height_norm)

        def point(index):
            if index >= len(kxy) or float(kconf[index]) < self.action_min_keypoint_conf:
                return None
            return (float(kxy[index][0]), float(kxy[index][1]))

        def midpoint(left, right):
            if left is None or right is None:
                return None
            return ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)

        left_shoulder = point(5)
        right_shoulder = point(6)
        left_wrist = point(9)
        right_wrist = point(10)
        left_hip = point(11)
        right_hip = point(12)
        left_knee = point(13)
        right_knee = point(14)
        left_ankle = point(15)
        right_ankle = point(16)
        shoulder_mid = midpoint(left_shoulder, right_shoulder)
        hip_mid = midpoint(left_hip, right_hip)
        knee_mid = midpoint(left_knee, right_knee)

        torso_verticality = None
        if shoulder_mid is not None and hip_mid is not None:
            torso_dx = hip_mid[0] - shoulder_mid[0]
            torso_dy = hip_mid[1] - shoulder_mid[1]
            torso_dist = math.hypot(torso_dx, torso_dy)
            if torso_dist > 1e-6:
                torso_verticality = abs(torso_dy) / torso_dist

        knee_angles = []
        left_knee_angle = self.angle_at(left_hip, left_knee, left_ankle) if left_hip and left_knee and left_ankle else None
        right_knee_angle = self.angle_at(right_hip, right_knee, right_ankle) if right_hip and right_knee and right_ankle else None
        if left_knee_angle is not None:
            knee_angles.append(left_knee_angle)
        if right_knee_angle is not None:
            knee_angles.append(right_knee_angle)
        knee_angle = self.median_value(knee_angles)

        left_wrist_above = bool(left_wrist and left_shoulder and left_wrist[1] < left_shoulder[1] - box_h * 0.05)
        right_wrist_above = bool(right_wrist and right_shoulder and right_wrist[1] < right_shoulder[1] - box_h * 0.05)
        left_wrist_x_norm = left_wrist[0] / max(1.0, float(width)) if left_wrist else None
        right_wrist_x_norm = right_wrist[0] / max(1.0, float(width)) if right_wrist else None

        torso_v = torso_verticality if torso_verticality is not None else 1.0
        posture_aspect = max(aspect, full_aspect)
        lying_like = (torso_v < 0.42) or (posture_aspect > 1.20 and torso_v < 0.68)

        knees_near_hips = False
        if hip_mid is not None and knee_mid is not None:
            knees_near_hips = abs(knee_mid[1] - hip_mid[1]) < box_h * 0.45
        bent_knee = knee_angle is not None and knee_angle < 150.0
        sitting_like = torso_v > 0.45 and (bent_knee or knees_near_hips) and not lying_like
        upright_like = torso_v > 0.65 and posture_aspect < 1.05

        return {
            "aspect": aspect,
            "full_aspect": full_aspect,
            "center_y_norm": center_y_norm,
            "full_center_y_norm": full_center_y_norm,
            "full_box_height_norm": full_box_height_norm,
            "torso_verticality": torso_verticality,
            "knee_angle": knee_angle,
            "left_wrist_above": left_wrist_above,
            "right_wrist_above": right_wrist_above,
            "left_wrist_x_norm": left_wrist_x_norm,
            "right_wrist_x_norm": right_wrist_x_norm,
            "lying_like": lying_like,
            "sitting_like": sitting_like,
            "upright_like": upright_like,
            "box_conf": pose["box_conf"],
            "valid_count": pose["valid_count"],
        }

    def predict_owner_pose_feature(self, image, source_meta=None):
        if image is None or image.size == 0 or not self.action_ready or self.action_pose_model is None:
            return None
        try:
            use_half = self.action_half and str(self.action_device).startswith("cuda")
            results = self.action_pose_model.predict(
                image,
                imgsz=self.action_imgsz,
                conf=self.action_conf,
                iou=self.action_iou,
                device=self.action_device,
                half=use_half,
                max_det=max(1, self.action_max_det),
                verbose=False,
            )
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Owner action pose inference failed: %s", exc)
            return None
        if not results:
            return None
        target_center = source_meta.get("pose_target_center_norm") if source_meta else None
        pose = self.select_owner_pose(results[0], image.shape[1], target_center=target_center)
        if pose is None:
            return None
        return self.build_pose_feature(pose, image.shape, source_meta=source_meta)

    def classify_owner_action(self, features):
        usable = [feature for feature in features if feature is not None]
        if not usable:
            return "unknown", 0.0, "no pose"
        if len(usable) < max(1, self.action_min_pose_samples):
            return "unknown", 0.15, "too few pose samples"

        sample_count = float(len(usable))
        first = usable[:max(1, len(usable) // 3)]
        last = usable[-max(1, len(usable) // 3):]

        first_center = self.median_value([feature["center_y_norm"] for feature in first])
        last_center = self.median_value([feature["center_y_norm"] for feature in last])
        first_full_center = self.median_value([feature.get("full_center_y_norm") for feature in first])
        last_full_center = self.median_value([feature.get("full_center_y_norm") for feature in last])
        first_torso = self.median_value([feature["torso_verticality"] for feature in first])
        last_torso = self.median_value([feature["torso_verticality"] for feature in last])
        first_aspect = self.median_value([feature.get("full_aspect", feature["aspect"]) for feature in first])
        last_aspect = self.median_value([feature.get("full_aspect", feature["aspect"]) for feature in last])
        first_height = self.median_value([feature.get("full_box_height_norm") for feature in first])
        last_height = self.median_value([feature.get("full_box_height_norm") for feature in last])
        first_lie_ratio = sum(1 for feature in first if feature["lying_like"]) / float(len(first))
        last_lie_ratio = sum(1 for feature in last if feature["lying_like"]) / float(len(last))
        first_upright_ratio = sum(1 for feature in first if feature["upright_like"]) / float(len(first))
        median_aspect = self.median_value([feature.get("full_aspect", feature["aspect"]) for feature in usable])
        median_torso = self.median_value([feature["torso_verticality"] for feature in usable])
        median_knee = self.median_value([feature["knee_angle"] for feature in usable])

        center_start = first_full_center if first_full_center is not None else first_center
        center_end = last_full_center if last_full_center is not None else last_center
        center_drop = (center_end - center_start) if center_start is not None and center_end is not None else 0.0
        torso_drop = (first_torso - last_torso) if first_torso is not None and last_torso is not None else 0.0
        aspect_gain = (last_aspect - first_aspect) if first_aspect is not None and last_aspect is not None else 0.0
        lie_ratio_gain = last_lie_ratio - first_lie_ratio
        height_ratio = (last_height / first_height) if first_height and last_height else None
        height_shrunk = height_ratio is not None and height_ratio < self.action_fall_height_shrink_ratio
        became_lying = (
            last_lie_ratio >= self.action_fall_late_lie_ratio
            and lie_ratio_gain >= self.action_fall_lie_ratio_gain
        )
        motion_signal = (
            center_drop > self.action_fall_center_drop
            or torso_drop > self.action_fall_torso_drop
            or aspect_gain > self.action_fall_aspect_gain
            or height_shrunk
        )
        plausible_start = first_upright_ratio >= self.action_fall_early_upright_ratio or first_lie_ratio <= 0.35

        rospy.loginfo(
            "Owner action fall metrics: samples=%d first_lie=%.2f last_lie=%.2f first_upright=%.2f "
            "center_drop=%.2f torso_drop=%.2f aspect_gain=%.2f height_ratio=%s",
            len(usable),
            first_lie_ratio,
            last_lie_ratio,
            first_upright_ratio,
            center_drop,
            torso_drop,
            aspect_gain,
            "%.2f" % height_ratio if height_ratio is not None else "NA",
        )
        if plausible_start and last_lie_ratio >= self.action_fall_late_lie_ratio and (motion_signal or became_lying):
            confidence = clamp(
                0.58
                + max(0.0, center_drop) * 1.4
                + max(0.0, torso_drop) * 0.5
                + max(0.0, aspect_gain) * 0.35
                + max(0.0, lie_ratio_gain) * 0.25,
                0.0,
                0.98,
            )
            return "falling", confidence, "fall transition"

        wave_scores = []
        for side in ("left", "right"):
            above_key = "%s_wrist_above" % side
            x_key = "%s_wrist_x_norm" % side
            wrist_x = [feature[x_key] for feature in usable if feature[above_key] and feature[x_key] is not None]
            if len(wrist_x) >= 3:
                x_range = max(wrist_x) - min(wrist_x)
                above_ratio = len(wrist_x) / sample_count
                if x_range > 0.11 and above_ratio >= 0.45:
                    wave_scores.append(clamp(0.50 + x_range * 2.0 + above_ratio * 0.25, 0.0, 0.95))
        if wave_scores:
            return "waving", max(wave_scores), "raised wrist motion"

        lie_ratio = sum(1 for feature in usable if feature["lying_like"]) / sample_count
        sit_ratio = sum(1 for feature in usable if feature["sitting_like"]) / sample_count
        required_ratio = clamp(self.action_static_required_ratio, 0.50, 0.95)
        rospy.loginfo(
            "Owner action feature summary: samples=%d lie_ratio=%.2f sit_ratio=%.2f aspect=%s torso=%s knee=%s",
            len(usable),
            lie_ratio,
            sit_ratio,
            "%.2f" % median_aspect if median_aspect is not None else "NA",
            "%.2f" % median_torso if median_torso is not None else "NA",
            "%.0f" % median_knee if median_knee is not None else "NA",
        )
        if lie_ratio >= required_ratio and (median_aspect is None or median_aspect > 1.05):
            return "lying", clamp(0.50 + lie_ratio * 0.45, 0.0, 0.95), "horizontal body posture"
        if sit_ratio >= required_ratio and median_torso is not None and median_torso > 0.45:
            return "sitting", clamp(0.50 + sit_ratio * 0.45, 0.0, 0.95), "bent seated posture"

        upright_ratio = sum(1 for feature in usable if feature["upright_like"]) / sample_count
        if self.action_report_standing and upright_ratio >= required_ratio:
            return "standing", clamp(0.45 + upright_ratio * 0.35, 0.0, 0.85), "upright posture"
        return "unknown", 0.25, "no reliable target action"

    def recognize_owner_action(self, owner_candidate):
        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        if not self.action_recognition_enabled:
            return "unknown", 0.0, "disabled"
        if not self.action_ready or self.action_pose_model is None:
            return "unknown", 0.0, "pose unavailable"

        self.stop_base()
        if self.action_pause_yolo:
            self.set_yolo_paused(True)
            rospy.sleep(0.15)

        features = []
        deadline = time.time() + max(0.5, self.action_sample_seconds)
        interval = 1.0 / max(1.0, self.action_sample_rate)
        next_sample = 0.0
        rospy.loginfo("Recognizing owner action for %.1fs with robot camera", self.action_sample_seconds)
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                now = time.time()
                if now >= next_sample:
                    image, source_meta = self.snapshot_owner_action_image()
                    feature = self.predict_owner_pose_feature(image, source_meta=source_meta)
                    if feature is not None:
                        features.append(feature)
                    next_sample = now + interval
                rospy.sleep(0.02)
        finally:
            if self.action_pause_yolo:
                self.set_yolo_paused(False)

        label, confidence, reason = self.classify_owner_action(features)
        rospy.loginfo(
            "Owner action verdict: label=%s confidence=%.2f reason=%s samples=%d",
            label,
            confidence,
            reason,
            len(features),
        )
        return label, confidence, reason

    @staticmethod
    def action_to_speech(label):
        messages = {
            "falling": "识别到主人摔倒。",
            "lying_ground": "识别到主人摔倒。",
            "waving": "主人正在挥手示意。",
            "lying": "识别到主人躺下。",
            "sitting": "主人当前坐着。",
            "standing": "主人当前站立，没有检测到指定异常动作。",
            "unknown": "我已识别到主人，但动作不确定。",
        }
        return messages.get(label, messages["unknown"])

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "image conversion failed: %s", exc)
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_time = time.time()

    def detections_callback(self, msg):
        with self.lock:
            self.latest_detections = list(msg.detections)
            self.latest_detections_time = time.time()

    def pointcloud_callback(self, msg):
        with self.lock:
            self.latest_pointcloud = msg
            self.latest_pointcloud_time = time.time()

    def scan_callback(self, msg):
        with self.lock:
            self.latest_scan = msg
            self.latest_scan_time = time.time()

    def odom_callback(self, msg):
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        position = msg.pose.pose.position
        with self.lock:
            self.latest_yaw = yaw
            self.latest_odom_xy = (float(position.x), float(position.y))
            self.latest_odom_time = time.time()

    def asr_callback(self, msg):
        text = str(msg.data).strip()
        if not text:
            return
        now = time.time()
        with self.lock:
            self.asr_sequence += 1
            self.latest_asr_text = text
            self.latest_asr_time = now
            self.asr_history.append({"sequence": self.asr_sequence, "time": now, "text": text})
            if len(self.asr_history) > 100:
                self.asr_history = self.asr_history[-100:]
        rospy.loginfo("ASR text received on %s: %s", self.asr_topic, text)

    def collect_asr_since(self, sequence_after):
        with self.lock:
            entries = [entry for entry in self.asr_history if entry["sequence"] > sequence_after]
        if not entries:
            return "", sequence_after

        parts = []
        for entry in entries:
            text = str(entry.get("text", "")).strip()
            if text:
                parts.append(text)
        transcript = " ".join(parts).strip()
        return transcript, int(entries[-1]["sequence"])

    @staticmethod
    def write_pcm_wav(path, raw_audio, sample_rate, channels):
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(int(channels))
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(raw_audio)

    def ensure_electrical_switch_asr_model(self):
        if self.electrical_switch_asr_ready and self.electrical_switch_asr_model is not None:
            return True

        if self.electrical_switch_asr_hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.electrical_switch_asr_hf_endpoint)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            rospy.logerr("faster-whisper is not available for direct switch ASR: %s", exc)
            return False

        model_path = self.electrical_switch_asr_model_path
        if os.path.sep in model_path and not os.path.exists(model_path):
            rospy.logerr("Direct switch ASR model path does not exist: %s", model_path)
            return False

        try:
            rospy.loginfo(
                "Loading direct switch ASR model: path=%s device=%s compute=%s",
                model_path,
                self.electrical_switch_asr_device,
                self.electrical_switch_asr_compute_type,
            )
            start = time.time()
            self.electrical_switch_asr_model = WhisperModel(
                model_path,
                device=self.electrical_switch_asr_device,
                compute_type=self.electrical_switch_asr_compute_type,
            )
            self.electrical_switch_asr_ready = True
            rospy.loginfo("Direct switch ASR model loaded in %.2fs", time.time() - start)
            return True
        except Exception as exc:
            self.electrical_switch_asr_model = None
            self.electrical_switch_asr_ready = False
            rospy.logerr("Failed to load direct switch ASR model: %s", exc)
            return False

    def record_electrical_switch_audio_window(self, window_seconds, window_index):
        sample_rate = max(8000, int(self.electrical_switch_asr_sample_rate))
        channels = max(1, int(self.electrical_switch_asr_channels))
        bytes_per_sample = 2
        target_seconds = max(0.5, float(window_seconds))
        arecord_seconds = max(1, int(math.ceil(target_seconds)))
        target_bytes = int(target_seconds * sample_rate * channels * bytes_per_sample)
        cmd = [
            "arecord",
            "-q",
            "-D",
            str(self.electrical_switch_asr_capture_device),
            "-d",
            str(arecord_seconds),
            "-f",
            "S16_LE",
            "-r",
            str(sample_rate),
            "-c",
            str(channels),
            "-t",
            "raw",
        ]

        rospy.loginfo(
            "Direct switch ASR window %d: recording %.1fs from ALSA device %s",
            window_index,
            target_seconds,
            self.electrical_switch_asr_capture_device,
        )
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=arecord_seconds + 4,
            )
        except subprocess.TimeoutExpired as exc:
            rospy.logerr("Direct switch ASR arecord timed out: %s", exc)
            return None
        except OSError as exc:
            rospy.logerr("Direct switch ASR failed to start arecord: %s", exc)
            return None

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            rospy.logerr("Direct switch ASR arecord failed: %s", stderr or "unknown error")
            return None

        raw_audio = proc.stdout[:target_bytes]
        if not raw_audio:
            rospy.logerr("Direct switch ASR arecord returned no audio")
            return None

        rms = audioop.rms(raw_audio, bytes_per_sample)
        peak = audioop.max(raw_audio, bytes_per_sample)
        rospy.loginfo("Direct switch ASR window %d audio stats: rms=%d peak=%d", window_index, rms, peak)

        try:
            os.makedirs(self.electrical_switch_asr_wav_dir, exist_ok=True)
            wav_path = os.path.join(
                self.electrical_switch_asr_wav_dir,
                "task1_switch_instruction_%03d.wav" % window_index,
            )
            self.write_pcm_wav(wav_path, raw_audio, sample_rate, channels)
            return wav_path
        except Exception as exc:
            rospy.logerr("Failed to write direct switch ASR wav: %s", exc)
            return None

    def transcribe_electrical_switch_audio(self, wav_path, window_index):
        if not self.ensure_electrical_switch_asr_model():
            return None

        try:
            start = time.time()
            segments, _ = self.electrical_switch_asr_model.transcribe(
                wav_path,
                language=self.electrical_switch_asr_language,
                vad_filter=self.electrical_switch_asr_vad_filter,
                beam_size=self.electrical_switch_asr_beam_size,
                condition_on_previous_text=False,
                no_speech_threshold=self.electrical_switch_asr_no_speech_threshold,
            )
            transcript = " ".join(segment.text.strip() for segment in segments).strip()
            rospy.loginfo(
                "Direct switch ASR window %d transcript: %s (%.2fs)",
                window_index,
                transcript or "<empty>",
                time.time() - start,
            )
            return transcript
        except Exception as exc:
            rospy.logerr("Direct switch ASR transcription failed: %s", exc)
            return None
        finally:
            if wav_path and not self.electrical_switch_asr_keep_wav:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def collect_direct_electrical_switch_transcript(self):
        if not self.ensure_electrical_switch_asr_model():
            return ""

        window_seconds = max(0.5, self.electrical_switch_instruction_window_seconds)
        max_empty_windows = max(0, self.electrical_switch_instruction_max_empty_windows)
        total_timeout = max(0.0, self.electrical_switch_instruction_timeout)
        deadline = time.time() + total_timeout if total_timeout > 0.0 else None
        empty_windows = 0
        window_index = 1

        while not rospy.is_shutdown():
            if deadline is not None and time.time() >= deadline:
                rospy.logwarn("Direct switch ASR wait timed out after %.1fs", total_timeout)
                break

            wav_path = self.record_electrical_switch_audio_window(window_seconds, window_index)
            if wav_path is None:
                break

            transcript = self.transcribe_electrical_switch_audio(wav_path, window_index)
            if transcript is None:
                break
            if transcript:
                return transcript

            empty_windows += 1
            rospy.logwarn(
                "Direct switch ASR window %d had no recognized speech; continuing with next %.1fs window",
                window_index,
                window_seconds,
            )
            if max_empty_windows > 0 and empty_windows >= max_empty_windows:
                rospy.logwarn("Direct switch ASR stopped after %d empty windows", empty_windows)
                break
            window_index += 1

        return ""

    def collect_ros_topic_electrical_switch_transcript(self):
        with self.lock:
            sequence_after_prompt = self.asr_sequence
            self.latest_asr_text = ""
            self.latest_asr_time = None

        window_seconds = max(0.5, self.electrical_switch_instruction_window_seconds)
        settle_seconds = max(0.0, self.electrical_switch_instruction_asr_settle_seconds)
        max_empty_windows = max(0, self.electrical_switch_instruction_max_empty_windows)
        total_timeout = max(0.0, self.electrical_switch_instruction_timeout)
        deadline = time.time() + total_timeout if total_timeout > 0.0 else None
        rate = rospy.Rate(10)
        empty_windows = 0
        window_index = 1

        while not rospy.is_shutdown():
            if deadline is not None and time.time() >= deadline:
                rospy.logwarn("Electrical switch topic-ASR wait timed out after %.1fs", total_timeout)
                break

            rospy.loginfo(
                "Electrical switch topic-ASR window %d: listening for %.1fs on %s",
                window_index,
                window_seconds,
                self.asr_topic,
            )
            window_end = time.time() + window_seconds
            while not rospy.is_shutdown() and time.time() < window_end:
                if deadline is not None and time.time() >= deadline:
                    break
                rate.sleep()

            if settle_seconds > 0.0:
                settle_end = time.time() + settle_seconds
                while not rospy.is_shutdown() and time.time() < settle_end:
                    rate.sleep()

            transcript, sequence_after_prompt = self.collect_asr_since(sequence_after_prompt)
            if transcript:
                rospy.loginfo("Electrical switch topic-ASR window %d transcript: %s", window_index, transcript)
                return transcript

            empty_windows += 1
            rospy.logwarn(
                "Electrical switch topic-ASR window %d had no ASR text; continuing with next %.1fs window",
                window_index,
                window_seconds,
            )
            if max_empty_windows > 0 and empty_windows >= max_empty_windows:
                rospy.logwarn("Electrical switch topic-ASR stopped after %d empty windows", empty_windows)
                break
            window_index += 1

        return ""

    def publish_electrical_switch_state(self):
        self.electrical_switch_state_pub.publish(String(data=self.electrical_switch_state))
        rospy.loginfo(
            "Electrical switch state: %s (topic=%s)",
            self.electrical_switch_state,
            self.electrical_switch_state_topic,
        )

    @staticmethod
    def parse_electrical_switch_action(content):
        """Normalize the short JSON response returned by the local LLM."""
        try:
            parsed = json.loads(content)
        except (TypeError, ValueError):
            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            if not match:
                return "unknown"
            try:
                parsed = json.loads(match.group(0))
            except (TypeError, ValueError):
                return "unknown"

        if not isinstance(parsed, dict):
            return "unknown"

        action_map = {
            "on": "on",
            "开": "on",
            "开启": "on",
            "打开": "on",
            "open": "on",
            "1": "on",
            "true": "on",
            "接通": "on",
            "上电": "on",
            "off": "off",
            "关": "off",
            "关闭": "off",
            "关掉": "off",
            "close": "off",
            "0": "off",
            "false": "off",
            "断开": "off",
            "断电": "off",
        }
        values = [parsed.get("action")]
        values.extend(value for value in parsed.values() if not isinstance(value, (dict, list)))
        for value in values:
            action = action_map.get(str(value).strip().lower())
            if action:
                return action
        return "unknown"

    @staticmethod
    def electrical_switch_keyword_fallback(transcript):
        """Conservative fallback used when Ollama is unavailable or malformed."""
        text = str(transcript).strip()
        negation = re.search(r"(不要|不用|别|不想|无需|不需要|禁止|不能|别把|不用把)", text)
        if re.search(r"(关|断开|断电|停|拔掉|灭)", text) and not re.search(r"(开|接通|上电|亮)", text):
            return "off"
        if (
            re.search(r"(开|接通|上电|启动|亮)", text)
            and not negation
            and not re.search(r"(关|断开|断电|灭)", text)
        ):
            return "on"
        return "unknown"

    def classify_electrical_switch_instruction(self, transcript):
        """Use local Ollama first, with the reference script's keyword fallback."""
        prompt = (
            "/no_think\n"
            "判断说话内容是否要开启或关闭电气开关。只输出一行 JSON，不要任何解释。\n"
            '{"action":"on"} 表示开启；{"action":"off"} 表示关闭；'
            '{"action":"unknown"} 表示与开关无关或无法判断。\n'
            '示例：“把灯打开” -> {"action":"on"}；“关掉电源” -> {"action":"off"}；'
            '“今天天气怎么样” -> {"action":"unknown"}\n'
            "说话内容：%s" % transcript
        )
        payload = {
            "model": self.electrical_switch_ollama_model,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.electrical_switch_ollama_keep_alive,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": 0,
                "top_p": 0.7,
                "num_predict": self.electrical_switch_ollama_max_tokens,
                "num_ctx": 1024,
            },
        }
        request = urllib.request.Request(
            self.electrical_switch_ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(1.0, self.electrical_switch_ollama_timeout)
            ) as response:
                raw = response.read().decode("utf-8")
            result = json.loads(raw)
            content = result.get("message", {}).get("content", "").strip()
            action = self.parse_electrical_switch_action(content)
            if action == "unknown":
                rospy.logwarn("Ollama returned an unknown electrical switch action: %s", content)
            else:
                rospy.loginfo("Electrical switch instruction classified by Ollama: %s", action)
            return action
        except (OSError, ValueError, TypeError, urllib.error.URLError) as exc:
            rospy.logwarn("Electrical switch Ollama classification failed: %s", exc)
            action = self.electrical_switch_keyword_fallback(transcript)
            rospy.loginfo("Using keyword fallback for electrical switch instruction: %s", action)
            return action

    def wait_for_electrical_switch_instruction(self):
        """Ask once, then process owner speech in repeated fixed windows."""
        if not self.electrical_switch_instruction_enabled:
            rospy.loginfo("Electrical switch voice interaction is disabled")
            return "unknown"

        source = self.electrical_switch_instruction_source
        if source in ("direct", "direct_asr", "mic", "microphone"):
            if not self.ensure_electrical_switch_asr_model():
                transcript = ""
            else:
                self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
                transcript = self.collect_direct_electrical_switch_transcript()
        elif source in ("topic", "ros", "ros_topic", "voice_topic"):
            with self.lock:
                self.latest_asr_text = ""
                self.latest_asr_time = None
            self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
            transcript = self.collect_ros_topic_electrical_switch_transcript()
        else:
            rospy.logwarn(
                "Unknown electrical_switch_instruction_source=%s; falling back to direct_asr",
                source,
            )
            if not self.ensure_electrical_switch_asr_model():
                transcript = ""
            else:
                self.say(self.electrical_switch_prompt, hold=self.electrical_switch_prompt_hold)
                transcript = self.collect_direct_electrical_switch_transcript()

        if not transcript:
            rospy.logwarn(
                "No electrical switch instruction text received by source=%s",
                source,
            )
            action = "unknown"
        else:
            rospy.loginfo("Electrical switch instruction transcript for LLM: %s", transcript)
            action = self.classify_electrical_switch_instruction(transcript)

        self.electrical_switch_state = action
        self.publish_electrical_switch_state()
        reply = {
            "on": self.electrical_switch_reply_on,
            "off": self.electrical_switch_reply_off,
            "unknown": self.electrical_switch_reply_unknown,
        }.get(action, self.electrical_switch_reply_unknown)
        self.say(reply)
        return action

    def get_latest_yaw(self):
        with self.lock:
            return self.latest_yaw

    def get_latest_odom_xy(self):
        with self.lock:
            return self.latest_odom_xy

    def wait_for_odom_xy(self, timeout=1.0):
        deadline = time.time() + max(0.0, timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            odom_xy = self.get_latest_odom_xy()
            if odom_xy is not None:
                return odom_xy
            rate.sleep()
        return self.get_latest_odom_xy()

    @staticmethod
    def xy_distance(start_xy, end_xy):
        if start_xy is None or end_xy is None:
            return None
        return math.hypot(float(end_xy[0]) - float(start_xy[0]), float(end_xy[1]) - float(start_xy[1]))

    def stop_base(self):
        self.cmd_pub.publish(Twist())

    def front_scan_distance(self):
        with self.lock:
            scan = self.latest_scan
            scan_time = self.latest_scan_time
        if scan is None or not scan.ranges:
            return None
        if scan_time is None or time.time() - scan_time > self.approach_scan_max_age:
            rospy.logwarn_throttle(2.0, "Front lidar data is stale during owner approach")
            return None

        half_window = math.radians(max(1.0, self.approach_front_scan_degrees))
        values = []
        for index, distance in enumerate(scan.ranges):
            angle = scan.angle_min + index * scan.angle_increment
            if abs(angle) <= half_window and math.isfinite(distance) and scan.range_min < distance < scan.range_max:
                values.append(float(distance))
        if not values:
            return None
        return min(values)

    def estimate_owner_position_from_pointcloud(self, det, image_width, image_height):
        with self.lock:
            cloud = self.latest_pointcloud
            cloud_time = self.latest_pointcloud_time
        if cloud is None:
            self.last_pointcloud_reason = "no point cloud on %s" % self.points_topic
            rospy.logwarn_throttle(2.0, "No Kinect point cloud received on %s", self.points_topic)
            return None
        if cloud.height <= 1 or cloud.width <= 1:
            self.last_pointcloud_reason = "point cloud is not organized"
            rospy.logwarn_throttle(2.0, "Point cloud on %s is not organized; cannot sample by image bbox", self.points_topic)
            return None

        age = time.time() - cloud_time if cloud_time is not None else None
        if age is not None and age > self.approach_pointcloud_max_age:
            self.last_pointcloud_reason = "point cloud is stale: age=%.2fs" % age
            rospy.logwarn_throttle(2.0, "Kinect point cloud is stale during owner approach: age=%.2fs", age)
            return None

        if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
            self.last_pointcloud_reason = "invalid image size for point cloud ROI"
            return None

        scale_x = float(cloud.width) / float(image_width)
        scale_y = float(cloud.height) / float(image_height)
        xmin = clamp(int(float(det.xmin) * scale_x), 0, int(cloud.width) - 1)
        xmax = clamp(int(float(det.xmax) * scale_x), xmin + 1, int(cloud.width))
        ymin = clamp(int(float(det.ymin) * scale_y), 0, int(cloud.height) - 1)
        ymax = clamp(int(float(det.ymax) * scale_y), ymin + 1, int(cloud.height))

        box_w = max(1, xmax - xmin)
        box_h = max(1, ymax - ymin)
        x_margin = int(box_w * clamp(self.approach_pointcloud_roi_x_margin, 0.0, 0.45))
        roi_x1 = clamp(xmin + x_margin, 0, int(cloud.width) - 1)
        roi_x2 = clamp(xmax - x_margin, roi_x1 + 1, int(cloud.width))

        y_min_ratio = clamp(self.approach_pointcloud_roi_y_min_ratio, 0.0, 0.95)
        y_max_ratio = clamp(self.approach_pointcloud_roi_y_max_ratio, y_min_ratio + 0.01, 1.0)
        roi_y1 = clamp(ymin + int(box_h * y_min_ratio), 0, int(cloud.height) - 1)
        roi_y2 = clamp(ymin + int(box_h * y_max_ratio), roi_y1 + 1, int(cloud.height))

        stride = max(1, self.approach_pointcloud_stride)
        uvs = [(u, v) for v in range(roi_y1, roi_y2, stride) for u in range(roi_x1, roi_x2, stride)]
        if not uvs:
            self.last_pointcloud_reason = "empty point cloud ROI"
            return None

        points = []
        try:
            for x, y, z in pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs):
                x = float(x)
                y = float(y)
                z = float(z)
                if all(math.isfinite(value) for value in (x, y, z)):
                    points.append((x, y, z))
        except Exception as exc:
            self.last_pointcloud_reason = "failed to read point cloud ROI: %s" % exc
            rospy.logwarn_throttle(2.0, "Failed to read owner ROI from point cloud: %s", exc)
            return None

        if len(points) < self.approach_pointcloud_min_samples:
            self.last_pointcloud_reason = "too few valid ROI points: %d < %d" % (
                len(points),
                self.approach_pointcloud_min_samples,
            )
            rospy.logwarn_throttle(
                2.0,
                "Owner point cloud ROI has too few valid points: %d < %d",
                len(points),
                self.approach_pointcloud_min_samples,
            )
            return None

        raw = np.asarray(points, dtype=np.float32)
        raw_x = float(np.median(raw[:, 0]))
        raw_y = float(np.median(raw[:, 1]))
        raw_z = float(np.median(raw[:, 2]))
        mode = self.approach_pointcloud_mode
        if mode not in ("auto", "optical", "base"):
            mode = "auto"

        frame = (cloud.header.frame_id or "").lower()
        horizontal = math.hypot(raw_x, raw_y)
        depth_dominant = raw_z > max(0.8, horizontal * 1.35)
        use_optical = mode == "optical" or (mode == "auto" and ("optical" in frame or depth_dominant))
        if use_optical:
            forward_values = raw[:, 2]
            lateral_values = -raw[:, 0]
            # Kinect optical frame uses +Y downward; estimate height above floor from camera height.
            height_values = float(self.lying_surface_camera_height) - raw[:, 1]
            used_mode = "optical"
        else:
            forward_values = raw[:, 0]
            lateral_values = raw[:, 1]
            height_values = raw[:, 2]
            used_mode = "base"

        valid = []
        for forward, lateral, height in zip(forward_values, lateral_values, height_values):
            forward = float(forward)
            lateral = float(lateral)
            height = float(height)
            if (
                self.approach_min_depth <= forward <= self.approach_max_depth
                and abs(lateral) <= self.approach_max_depth
                and math.isfinite(height)
                and -0.20 <= height <= 2.00
            ):
                valid.append((forward, lateral, height))
        if len(valid) < self.approach_pointcloud_min_samples:
            self.last_pointcloud_reason = "too few in-range ROI points: %d < %d mode=%s raw=(%.2f, %.2f, %.2f)" % (
                len(valid),
                self.approach_pointcloud_min_samples,
                used_mode,
                raw_x,
                raw_y,
                raw_z,
            )
            rospy.logwarn_throttle(
                2.0,
                "Owner point cloud ROI has too few in-range points: %d < %d mode=%s raw=(%.2f, %.2f, %.2f)",
                len(valid),
                self.approach_pointcloud_min_samples,
                used_mode,
                raw_x,
                raw_y,
                raw_z,
            )
            return None

        valid = np.asarray(valid, dtype=np.float32)
        depth_percentile = clamp(self.approach_pointcloud_depth_percentile, 5.0, 50.0)
        forward_surface = float(np.percentile(valid[:, 0], depth_percentile))
        surface_band = max(0.05, self.approach_pointcloud_surface_band)
        surface = valid[valid[:, 0] <= forward_surface + surface_band]
        if len(surface) >= self.approach_pointcloud_min_samples:
            valid = surface

        person_x = float(np.median(valid[:, 0]))
        person_y = float(np.median(valid[:, 1]))
        surface_height_median = float(np.median(valid[:, 2]))
        surface_height_p20 = float(np.percentile(valid[:, 2], 20.0))
        surface_height_p80 = float(np.percentile(valid[:, 2], 80.0))
        distance = math.hypot(person_x, person_y)
        bearing = math.atan2(person_y, max(0.05, person_x))
        self.last_pointcloud_reason = ""
        return {
            "x": person_x,
            "y": person_y,
            "distance": distance,
            "bearing": bearing,
            "surface_height_median": surface_height_median,
            "surface_height_p20": surface_height_p20,
            "surface_height_p80": surface_height_p80,
            "mode": used_mode,
            "samples": int(len(valid)),
            "frame": cloud.header.frame_id,
        }

    def set_yolo_paused(self, paused):
        self.pause_yolo_pub.publish(Bool(data=bool(paused)))

    def wait_for_tts_subscriber(self):
        if not self.say_wait_for_subscribers:
            return True
        deadline = time.time() + self.say_wait_timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.say_pub.get_num_connections() > 0:
                return True
            time.sleep(0.05)
        rospy.logwarn("No subscribers connected to %s; TTS message may not be spoken", self.say_topic)
        return False

    def say(self, text, hold=None):
        rospy.loginfo("TTS: %s", text)
        self.wait_for_tts_subscriber()
        for index in range(self.say_repeat_count):
            self.say_pub.publish(String(data=text))
            if index + 1 < self.say_repeat_count and self.say_repeat_interval > 0:
                rospy.sleep(self.say_repeat_interval)
        delay = self.say_after_publish_delay if hold is None else float(hold)
        if delay > 0:
            rospy.sleep(delay)

    def publish_manipulator_command(self, lift, gripper):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ["lift", "gripper"]
        msg.position = [float(lift), float(gripper)]
        msg.velocity = [float(self.fall_assist_arm_lift_velocity), float(self.fall_assist_arm_gripper_velocity)]
        self.mani_ctrl_pub.publish(msg)

    def hold_manipulator_command(self, lift, gripper, seconds):
        duration = max(0.0, float(seconds))
        rate_hz = max(0.5, float(self.fall_assist_arm_command_rate))
        deadline = time.time() + duration
        rate = rospy.Rate(rate_hz)
        self.publish_manipulator_command(lift, gripper)
        while not rospy.is_shutdown() and time.time() < deadline:
            self.publish_manipulator_command(lift, gripper)
            rate.sleep()

    def perform_fall_assist_arm_motion(self):
        if not self.fall_assist_arm_enabled:
            rospy.loginfo("Fall assist arm motion is disabled")
            return True

        rospy.loginfo(
            "Fall assist arm motion: extend lift=%.2f gripper=%.2f hold=%.1fs then retract lift=%.2f gripper=%.2f topic=%s",
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_hold_seconds,
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
            self.mani_ctrl_topic,
        )

        self.hold_manipulator_command(
            self.fall_assist_arm_extend_lift,
            self.fall_assist_arm_extend_gripper,
            self.fall_assist_arm_extend_wait + self.fall_assist_arm_hold_seconds,
        )
        self.hold_manipulator_command(
            self.fall_assist_arm_retract_lift,
            self.fall_assist_arm_retract_gripper,
            self.fall_assist_arm_retract_wait,
        )
        return True

    def wait_for_hardware_inputs(self):
        deadline = time.time() + max(1.0, self.hardware_check_timeout)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            with self.lock:
                has_image = self.latest_image_time is not None
                has_scan = self.latest_scan_time is not None
                has_odom = self.latest_odom_time is not None

            image_ok = has_image or not self.require_kinect
            scan_ok = has_scan or not self.require_lidar
            odom_ok = has_odom or not self.require_odom
            if image_ok and scan_ok and odom_ok:
                rospy.loginfo(
                    "Real robot inputs ready: image=%s scan=%s odom=%s",
                    has_image,
                    has_scan,
                    has_odom,
                )
                return True
            rospy.loginfo_throttle(
                2.0,
                "Waiting for real robot inputs: image=%s scan=%s odom=%s",
                has_image,
                has_scan,
                has_odom,
            )
            rate.sleep()

        missing = []
        details = []
        with self.lock:
            if self.require_kinect and self.latest_image_time is None:
                missing.append(self.image_topic)
                details.append(self.describe_missing_topic(self.image_topic, "Kinect color image"))
            if self.require_lidar and self.latest_scan_time is None:
                missing.append(self.scan_topic)
                details.append(self.describe_missing_topic(self.scan_topic, "lidar scan"))
            if self.require_odom and self.latest_odom_time is None:
                missing.append(self.odom_topic)
                details.append(self.describe_missing_topic(self.odom_topic, "wheel odometry"))

        message = "required real robot inputs are not receiving messages: %s" % ", ".join(missing)
        if details:
            message += "; " + "; ".join(details)
        raise MissingHardwareError(message)

    def is_topic_advertised(self, topic_name):
        try:
            published_topics = rospy.get_published_topics("/")
        except Exception as exc:
            rospy.logwarn("Unable to query published topics while checking %s: %s", topic_name, exc)
            return False

        normalized = topic_name.rstrip("/") or "/"
        for name, _topic_type in published_topics:
            if (name.rstrip("/") or "/") == normalized:
                return True
        return False

    def describe_missing_topic(self, topic_name, label):
        if self.is_topic_advertised(topic_name):
            return "%s topic %s is advertised but produced no messages; check `rostopic hz %s`" % (
                label,
                topic_name,
                topic_name,
            )
        return "%s topic %s is not advertised; check the corresponding driver launch" % (label, topic_name)

    def load_waypoint_pose(self):
        if not os.path.exists(self.waypoint_file):
            raise RuntimeError("waypoint file not found: %s" % self.waypoint_file)

        root = ET.parse(self.waypoint_file).getroot()
        for waypoint in root.findall("Waypoint"):
            name = waypoint.findtext("Name", "")
            if name != self.waypoint_name:
                continue

            pose = Pose()
            pose.position.x = float(waypoint.findtext("Pos_x", "0"))
            pose.position.y = float(waypoint.findtext("Pos_y", "0"))
            pose.position.z = float(waypoint.findtext("Pos_z", "0"))
            pose.orientation.x = float(waypoint.findtext("Ori_x", "0"))
            pose.orientation.y = float(waypoint.findtext("Ori_y", "0"))
            pose.orientation.z = float(waypoint.findtext("Ori_z", "0"))
            pose.orientation.w = float(waypoint.findtext("Ori_w", "1"))
            return pose

        raise RuntimeError("waypoint not found: %s in %s" % (self.waypoint_name, self.waypoint_file))

    def clear_move_base_costmaps(self, reason):
        try:
            rospy.loginfo("Clearing move_base costmaps %s", reason)
            rospy.wait_for_service(self.clear_costmaps_service, timeout=self.clear_costmaps_timeout)
            clear_costmaps = rospy.ServiceProxy(self.clear_costmaps_service, Empty)
            clear_costmaps()
            rospy.sleep(0.3)
            return True
        except Exception as exc:
            rospy.logwarn("Unable to clear move_base costmaps %s: %s", reason, exc)
            return False

    def send_navigation_goal(self, goal):
        self.move_base.send_goal(goal)
        finished = self.move_base.wait_for_result(rospy.Duration(self.navigate_timeout))
        if not finished:
            self.move_base.cancel_goal()
            return False, "move_base timed out while navigating to %s" % self.waypoint_name

        state = self.move_base.get_state()
        if state != GoalStatus.SUCCEEDED:
            return False, "move_base failed with state %s" % state
        return True, ""

    def navigate_to_waypoint(self):
        if not self.navigate_enabled:
            rospy.loginfo("Navigation disabled; assuming robot is already at %s", self.waypoint_name)
            return

        rospy.loginfo("Waiting for move_base action server")
        if not self.move_base.wait_for_server(rospy.Duration(25.0)):
            raise RuntimeError("move_base action server is not available")

        if self.clear_costmaps_before_navigation:
            self.clear_move_base_costmaps("before navigating to %s" % self.waypoint_name)

        pose = self.load_waypoint_pose()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = pose

        rospy.loginfo(
            "Navigating real robot to waypoint %s: x=%.3f y=%.3f",
            self.waypoint_name,
            pose.position.x,
            pose.position.y,
        )

        success, error_message = self.send_navigation_goal(goal)
        if not success and self.retry_navigation_after_clear:
            rospy.logwarn("%s; clearing costmaps and retrying once", error_message)
            self.clear_move_base_costmaps("after navigation failure")
            goal.target_pose.header.stamp = rospy.Time.now()
            success, error_message = self.send_navigation_goal(goal)

        if not success:
            raise RuntimeError(error_message)
        rospy.loginfo("Arrived at waypoint %s", self.waypoint_name)

    def snapshot_candidates(self):
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return []
            image = self.latest_image.copy()
            detections = list(self.latest_detections)

        height, width = image.shape[:2]
        candidates = []
        for det in detections:
            class_name = getattr(det, "class_name", "")
            if class_name and class_name != "person":
                continue
            if float(det.score) < self.detection_min_score:
                continue

            xmin = clamp(int(det.xmin), 0, width - 1)
            ymin = clamp(int(det.ymin), 0, height - 1)
            xmax = clamp(int(det.xmax), 0, width - 1)
            ymax = clamp(int(det.ymax), 0, height - 1)
            if xmax <= xmin or ymax <= ymin:
                continue

            box_w = xmax - xmin
            box_h = ymax - ymin
            area_ratio = float(box_w * box_h) / float(width * height)
            if area_ratio < self.detection_min_area_ratio:
                continue

            face_crop_padding = clamp(float(self.face_crop_padding), 0.05, 0.40)
            pad_x = int(box_w * face_crop_padding)
            pad_y = int(box_h * face_crop_padding)
            cx1 = clamp(xmin - pad_x, 0, width - 1)
            cy1 = clamp(ymin - pad_y, 0, height - 1)
            cx2 = clamp(xmax + pad_x, 0, width - 1)
            cy2 = clamp(ymax + pad_y, 0, height - 1)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            center_error = abs((float(det.center_x) / float(width)) - 0.5)
            priority = float(det.score) + area_ratio * 4.0 - center_error * 0.2
            candidates.append({
                "det": det,
                "crop": crop,
                "score": float(det.score),
                "area_ratio": area_ratio,
                "priority": priority,
                "image_width": width,
                "image_height": height,
            })

        candidates.sort(key=lambda item: item["priority"], reverse=True)
        return candidates[:self.verify_top_k]

    def crop_candidate_face_regions(self, candidate, include_extra_side_ratios=True):
        crop = candidate.get("crop")
        if crop is None or crop.size == 0:
            return []
        height, width = crop.shape[:2]
        top_ratio = clamp(float(self.face_crop_top_ratio), 0.35, 1.0)
        side_ratios = [float(self.face_crop_lying_side_ratio)]
        if include_extra_side_ratios:
            side_ratios.extend(float(ratio) for ratio in self.face_crop_lying_extra_side_ratios)

        det = candidate.get("det")
        if det is not None:
            person_width = max(1.0, float(det.xmax) - float(det.xmin))
            person_height = max(1.0, float(det.ymax) - float(det.ymin))
        else:
            person_width = float(width)
            person_height = float(height)

        regions = []
        seen = set()

        def add_region(name, x1, y1, x2, y2, allow_rotations):
            x1 = clamp(int(x1), 0, width - 1)
            y1 = clamp(int(y1), 0, height - 1)
            x2 = clamp(int(x2), x1 + 1, width)
            y2 = clamp(int(y2), y1 + 1, height)
            if x2 - x1 < 8 or y2 - y1 < 8:
                return
            key = (x1, y1, x2, y2)
            if key in seen:
                return
            seen.add(key)
            regions.append((name, crop[y1:y2, x1:x2], bool(allow_rotations)))

        if person_height >= person_width:
            upper_y2 = max(1, int(height * top_ratio))
            add_region("upper", 0, 0, width, upper_y2, allow_rotations=False)
        else:
            for raw_ratio in side_ratios:
                side_ratio = clamp(float(raw_ratio), 0.35, 0.75)
                side_width = max(1, int(width * side_ratio))
                ratio_name = int(round(side_ratio * 100.0))
                add_region("left_side_%d" % ratio_name, 0, 0, side_width, height, allow_rotations=True)
                add_region("right_side_%d" % ratio_name, width - side_width, 0, width, height, allow_rotations=True)

        return regions

    def face_image_variants(self, region_name, face_image, enhanced=False):
        if not enhanced or not self.face_candidate_enhance:
            yield region_name, face_image
            return

        try:
            lab_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
            lightness, channel_a, channel_b = cv2.split(lab_image)
            lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
            yield region_name + ":clahe", cv2.cvtColor(
                cv2.merge((lightness, channel_a, channel_b)),
                cv2.COLOR_LAB2BGR,
            )
        except Exception:
            pass

        yield region_name + ":up1.5", cv2.resize(
            face_image,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC,
        )

    def best_owner_face_match(self, candidate_embedding, owner_embeddings):
        return max(
            (
                (float(np.dot(item["embedding"], candidate_embedding)), item.get("path", "owner"))
                for item in owner_embeddings
            ),
            key=lambda value: value[0],
        )

    def face_region_orientations(self, region_name, face_image, allow_rotations):
        yield region_name, face_image
        if not allow_rotations or not self.face_crop_try_rotations:
            return
        yield region_name + ":rot90", cv2.rotate(face_image, cv2.ROTATE_90_CLOCKWISE)
        yield region_name + ":rot270", cv2.rotate(face_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yield region_name + ":rot180", cv2.rotate(face_image, cv2.ROTATE_180)

    def verify_candidate_with_face(self, candidate):
        if not self.face_ready or self.face_app is None:
            return None, 0.0, "face unavailable"
        owner_embeddings = list(getattr(self, "owner_face_embeddings", []))
        if not owner_embeddings and self.owner_face_embedding is not None:
            owner_embeddings = [{"path": self.owner_image_path, "embedding": self.owner_face_embedding}]
        if not owner_embeddings:
            return None, 0.0, "face unavailable"

        face_regions = self.crop_candidate_face_regions(candidate)
        if not face_regions:
            return None, 0.0, "empty face crop"

        best_similarity = None
        best_reference_path = None
        best_region_name = None
        face_count = 0
        elapsed_ms = 0.0
        had_error = False

        def run_regions(regions, enhancement_passes):
            nonlocal best_similarity, best_reference_path, best_region_name
            nonlocal face_count, elapsed_ms, had_error
            for enhanced in enhancement_passes:
                for region_name, face_image, allow_rotations in regions:
                    if enhanced and not allow_rotations:
                        continue
                    for variant_name, variant_image in self.face_image_variants(region_name, face_image, enhanced):
                        for oriented_name, oriented_image in self.face_region_orientations(
                            variant_name,
                            variant_image,
                            allow_rotations,
                        ):
                            try:
                                start = time.time()
                                faces = self.face_app.get(oriented_image)
                                elapsed_ms += (time.time() - start) * 1000.0
                            except Exception as exc:
                                had_error = True
                                rospy.logwarn("Face verification failed on %s crop: %s", oriented_name, exc)
                                continue

                            for candidate_face in faces or []:
                                face_count += 1
                                candidate_embedding = self.normalize_embedding(candidate_face.embedding)
                                if candidate_embedding is None:
                                    continue
                                similarity, reference_path = self.best_owner_face_match(
                                    candidate_embedding,
                                    owner_embeddings,
                                )
                                if best_similarity is None or similarity > best_similarity:
                                    best_similarity = similarity
                                    best_reference_path = reference_path
                                    best_region_name = oriented_name
                            if best_similarity is not None and best_similarity >= self.face_accept_threshold:
                                return True
            return False

        # Most non-owners are far below the threshold in the primary side crop.
        # Keep the wider/enhanced search for borderline faces so low-confidence
        # owner views retain the same fallback behavior as before.
        if self.face_fast_pass_enabled:
            fast_regions = self.crop_candidate_face_regions(candidate, include_extra_side_ratios=False)
            accepted = run_regions(fast_regions, [False])
            if accepted:
                rospy.loginfo(
                    "Face owner verdict: accepted by fast pass similarity=%.3f crop=%s faces=%d elapsed=%.1fms",
                    best_similarity,
                    best_region_name,
                    face_count,
                    elapsed_ms,
                )
                return True, best_similarity, "face accepted"
            if (
                best_similarity is not None
                and not had_error
                and best_similarity <= self.face_fast_pass_reject_threshold
            ):
                rospy.loginfo(
                    "Face owner verdict: rejected by fast pass similarity=%.3f crop=%s faces=%d elapsed=%.1fms",
                    best_similarity,
                    best_region_name,
                    face_count,
                    elapsed_ms,
                )
                return False, best_similarity, "face rejected by fast pass"

        if self.face_fast_pass_enabled:
            # The primary regions have already been evaluated above. Continue
            # with only the remaining raw regions, then enhance all eligible
            # side regions so borderline owner views keep the old fallback.
            if run_regions(face_regions[len(fast_regions):], [False]):
                return True, best_similarity, "face accepted"
            if self.face_candidate_enhance:
                if run_regions(face_regions, [True]):
                    return True, best_similarity, "face accepted"
        else:
            enhancement_passes = [False]
            if self.face_candidate_enhance:
                enhancement_passes.append(True)
            run_regions(face_regions, enhancement_passes)

        if best_similarity is None:
            if had_error:
                return None, 0.0, "face verification error"
            rospy.loginfo("Face verifier saw no face in candidate crops")
            return None, 0.0, "no face"

        similarity = best_similarity
        reference_path = best_reference_path or self.owner_image_path
        reference_name = os.path.basename(reference_path)
        if similarity >= self.face_accept_threshold:
            rospy.loginfo(
                "Face owner verdict: accepted similarity=%.3f reference=%s crop=%s faces=%d elapsed=%.1fms",
                similarity,
                reference_name,
                best_region_name,
                face_count,
                elapsed_ms,
            )
            return True, similarity, "face accepted"
        if self.face_fast_reject and similarity <= self.face_reject_threshold:
            rospy.loginfo(
                "Face owner verdict: rejected similarity=%.3f best_reference=%s crop=%s faces=%d elapsed=%.1fms",
                similarity,
                reference_name,
                best_region_name,
                face_count,
                elapsed_ms,
            )
            return False, similarity, "face rejected"

        rospy.loginfo(
            "Face owner verdict: uncertain similarity=%.3f best_reference=%s crop=%s faces=%d elapsed=%.1fms",
            similarity,
            reference_name,
            best_region_name,
            face_count,
            elapsed_ms,
        )
        return None, similarity, "face uncertain"

    def verify_candidate(self, candidate):
        decision, score, reason = self.verify_candidate_with_face(candidate)
        if decision is True:
            return True, score, reason
        if decision is False:
            return False, score, reason
        if self.allow_unverified_owner:
            rospy.logwarn("Accepting unverified owner candidate for hardware debug only: %s", reason)
            return True, float(candidate.get("score", 0.0)), "unverified debug accept"
        return False, score, reason

    def wait_for_odom_yaw(self, timeout=3.0):
        deadline = time.time() + timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            yaw = self.get_latest_yaw()
            if yaw is not None:
                return yaw
            rate.sleep()
        return None

    def rotate_to_yaw(self, target_yaw, timeout=18.0):
        deadline = time.time() + timeout
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() < deadline:
            current_yaw = self.get_latest_yaw()
            if current_yaw is None:
                return False

            error = signed_angle_diff(target_yaw, current_yaw)
            if abs(error) < 0.08:
                self.stop_base()
                return True

            twist = Twist()
            twist.angular.z = clamp(1.0 * error, -self.return_angular_speed, self.return_angular_speed)
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        return False

    def merge_scan_candidates(self, pool, candidates, yaw):
        for candidate in candidates:
            item = dict(candidate)
            item["yaw"] = yaw
            item["priority"] = float(item.get("priority", 0.0))

            replaced = False
            for index, existing in enumerate(pool):
                existing_yaw = existing.get("yaw")
                same_view = yaw is not None and existing_yaw is not None and abs(signed_angle_diff(yaw, existing_yaw)) < 0.35
                same_center = abs(
                    (float(item["det"].center_x) / float(item["image_width"]))
                    - (float(existing["det"].center_x) / float(existing["image_width"]))
                ) < 0.15
                if same_view and same_center:
                    if item["priority"] > existing.get("priority", 0.0):
                        pool[index] = item
                    replaced = True
                    break
            if not replaced:
                pool.append(item)

        pool.sort(key=lambda value: value.get("priority", 0.0), reverse=True)
        return pool[:self.scan_candidate_pool_size]

    def verify_candidate_pool(self, pool):
        if not pool:
            return None, None

        for candidate in sorted(pool, key=lambda value: value.get("priority", 0.0), reverse=True)[:self.verify_after_scan_top_k]:
            accepted, confidence, reason = self.verify_candidate(candidate)
            if accepted:
                det = candidate["det"]
                self.owner_track_center = float(det.center_x) / float(candidate["image_width"])
                rospy.loginfo(
                    "Owner accepted at bbox=(%d,%d,%d,%d), confidence=%.2f reason=%s",
                    det.xmin,
                    det.ymin,
                    det.xmax,
                    det.ymax,
                    confidence,
                    reason,
                )
                return candidate, candidate.get("yaw")
        return None, None

    def scan_for_owner_by_time(self):
        deadline = time.time() + self.scan_duration
        last_verify_time = 0.0
        last_collect_time = 0.0
        owner_candidate = None
        owner_yaw = None
        scan_candidate_pool = []
        rate = rospy.Rate(10)

        rospy.logwarn("No odom yaw received on %s; falling back to timed scan", self.odom_topic)
        while not rospy.is_shutdown() and time.time() < deadline:
            twist = Twist()
            twist.angular.z = self.scan_angular_speed
            self.cmd_pub.publish(twist)

            now = time.time()
            if now - last_collect_time >= self.candidate_collection_interval:
                candidates = self.snapshot_candidates()
                if candidates:
                    scan_candidate_pool = self.merge_scan_candidates(scan_candidate_pool, candidates, self.get_latest_yaw())
                last_collect_time = now

            if self.verify_during_scan and owner_candidate is None and now - last_verify_time >= self.candidate_cooldown:
                candidates = self.snapshot_candidates()
                if candidates:
                    self.stop_base()
                    rospy.sleep(0.2)
                    current_pool = self.merge_scan_candidates([], candidates, self.get_latest_yaw())
                    owner_candidate, owner_yaw = self.verify_candidate_pool(current_pool)
                    last_verify_time = time.time()
                    if owner_candidate is not None:
                        return owner_candidate
            rate.sleep()

        self.stop_base()
        if owner_candidate is None:
            owner_candidate, owner_yaw = self.verify_candidate_pool(scan_candidate_pool)
        if owner_candidate is not None and owner_yaw is not None and self.scan_return_to_owner:
            self.rotate_to_yaw(owner_yaw)
        return owner_candidate

    def scan_for_owner(self):
        rospy.sleep(self.scan_after_arrival_delay)
        start_yaw = self.wait_for_odom_yaw()
        if start_yaw is None:
            return self.scan_for_owner_by_time()

        deadline = time.time() + self.scan_timeout
        last_verify_time = 0.0
        last_yaw = start_yaw
        rotated_angle = 0.0
        owner_candidate = None
        owner_yaw = None
        last_collect_time = 0.0
        scan_candidate_pool = []
        rate = rospy.Rate(10)

        rospy.loginfo(
            "Scanning %.2f rad at %.2f rad/s using odom yaw from %s",
            self.scan_total_angle,
            self.scan_angular_speed,
            self.odom_topic,
        )
        while not rospy.is_shutdown() and rotated_angle < self.scan_total_angle and time.time() < deadline:
            twist = Twist()
            twist.angular.z = self.scan_angular_speed
            self.cmd_pub.publish(twist)

            current_yaw = self.get_latest_yaw()
            if current_yaw is not None:
                delta = signed_angle_diff(current_yaw, last_yaw)
                if abs(delta) < 0.7:
                    rotated_angle += abs(delta)
                last_yaw = current_yaw

            now = time.time()
            rospy.loginfo_throttle(
                3.0,
                "Owner scan rotated %.0f / %.0f deg",
                math.degrees(rotated_angle),
                math.degrees(self.scan_total_angle),
            )
            if now - last_collect_time >= self.candidate_collection_interval:
                candidates = self.snapshot_candidates()
                if candidates:
                    scan_candidate_pool = self.merge_scan_candidates(scan_candidate_pool, candidates, current_yaw)
                last_collect_time = now

            if self.verify_during_scan and owner_candidate is None and now - last_verify_time >= self.candidate_cooldown:
                candidates = self.snapshot_candidates()
                if candidates:
                    self.stop_base()
                    rospy.sleep(0.2)
                    current_pool = self.merge_scan_candidates([], candidates, current_yaw)
                    owner_candidate, owner_yaw = self.verify_candidate_pool(current_pool)
                    last_verify_time = time.time()
                    if owner_candidate is not None:
                        return owner_candidate
            rate.sleep()

        self.stop_base()
        if rotated_angle < self.scan_total_angle:
            rospy.logwarn(
                "Owner scan stopped before full angle: %.0f / %.0f deg",
                math.degrees(rotated_angle),
                math.degrees(self.scan_total_angle),
            )

        if owner_candidate is None:
            rospy.loginfo(
                "Scan collected %d candidate views; verifying top %d",
                len(scan_candidate_pool),
                self.verify_after_scan_top_k,
            )
            owner_candidate, owner_yaw = self.verify_candidate_pool(scan_candidate_pool)

        if owner_candidate is not None and owner_yaw is not None and self.scan_return_to_owner:
            rospy.loginfo("Scan complete; rotating back to owner heading")
            self.rotate_to_yaw(owner_yaw)
        return owner_candidate

    def select_tracking_detection(self):
        with self.lock:
            if self.latest_image is None or not self.latest_detections:
                return None, None, None
            height, width = self.latest_image.shape[:2]
            detections = list(self.latest_detections)

        best = None
        best_score = -1.0
        for det in detections:
            class_name = getattr(det, "class_name", "")
            if class_name and class_name != "person":
                continue
            if float(det.score) < self.detection_min_score:
                continue
            center_norm = float(det.center_x) / float(width)
            area_ratio = max(0.0, float((det.xmax - det.xmin) * (det.ymax - det.ymin)) / float(width * height))
            if self.owner_track_center is None:
                tracking_penalty = abs(center_norm - 0.5)
            else:
                tracking_penalty = abs(center_norm - self.owner_track_center)
            score = float(det.score) + area_ratio * 5.0 - tracking_penalty * 1.4
            if score > best_score:
                best = det
                best_score = score
        return best, width, height

    def center_owner_in_camera(self, owner_candidate=None):
        if not self.center_owner_enabled:
            return True

        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        rospy.loginfo("Centering verified owner in real robot camera")
        deadline = time.time() + max(0.5, self.center_owner_timeout)
        centered_count = 0
        rate = rospy.Rate(12)
        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, _height = self.select_tracking_detection()
            twist = Twist()

            if det is None or width is None or width <= 0:
                last_center = self.owner_track_center if self.owner_track_center is not None else 0.5
                direction = 1.0 if last_center > 0.5 else -1.0
                twist.angular.z = -direction * self.center_owner_lost_turn_speed
                self.cmd_pub.publish(twist)
                centered_count = 0
                rate.sleep()
                continue

            center_norm = float(det.center_x) / float(width)
            self.owner_track_center = center_norm
            x_error = center_norm - 0.5
            if abs(x_error) <= self.center_owner_tolerance:
                self.stop_base()
                centered_count += 1
                if centered_count >= 3:
                    rospy.loginfo("Owner centered in camera: center=%.3f", center_norm)
                    return True
                rate.sleep()
                continue

            centered_count = 0
            twist.angular.z = clamp(
                -self.center_owner_angular_gain * x_error,
                -self.center_owner_max_angular_speed,
                self.center_owner_max_angular_speed,
            )
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.stop_base()
        rospy.logwarn("Owner centering timed out")
        return False

    @staticmethod
    def owner_detection_image_context(det, image_width, image_height):
        width = max(1.0, float(image_width))
        height = max(1.0, float(image_height))
        xmin = float(det.xmin)
        xmax = float(det.xmax)
        ymin = float(det.ymin)
        ymax = float(det.ymax)
        box_w = max(1.0, xmax - xmin)
        box_h = max(1.0, ymax - ymin)
        center_y = float(getattr(det, "center_y", (ymin + ymax) * 0.5))
        return {
            "bbox_bottom_norm": clamp(ymax / height, 0.0, 1.5),
            "bbox_center_y_norm": clamp(center_y / height, 0.0, 1.5),
            "bbox_aspect": box_w / box_h,
            "bbox_height_norm": box_h / height,
            "bbox_center_x_norm": clamp(float(det.center_x) / width, 0.0, 1.5),
        }

    def capture_fallen_owner_position(self, owner_candidate=None):
        deadline = time.time() + max(0.1, self.fall_approach_position_sample_seconds)
        positions = []
        fallback_used = False
        image_context = None
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, height = self.select_tracking_detection()
            if (det is None or width is None or height is None) and owner_candidate is not None and not positions:
                det = owner_candidate.get("det")
                width = owner_candidate.get("image_width")
                height = owner_candidate.get("image_height")
                fallback_used = det is not None

            if det is None or width is None or height is None or width <= 0 or height <= 0:
                self.last_approach_failure_reason = "no owner detection before fallen-owner approach"
                rate.sleep()
                continue

            self.owner_track_center = float(det.center_x) / float(width)
            image_context = self.owner_detection_image_context(det, width, height)
            position = self.estimate_owner_position_from_pointcloud(det, width, height)
            if position is not None:
                positions.append(position)
                if len(positions) >= self.fall_approach_min_position_samples:
                    break
            else:
                self.last_approach_failure_reason = self.last_pointcloud_reason or "no valid fallen-owner point cloud"
            rate.sleep()

        if not positions:
            if not self.last_approach_failure_reason:
                self.last_approach_failure_reason = "no valid fallen-owner position before blind approach"
            return None

        xs = np.asarray([position["x"] for position in positions], dtype=np.float32)
        ys = np.asarray([position["y"] for position in positions], dtype=np.float32)
        surface_heights = np.asarray(
            [position["surface_height_median"] for position in positions if "surface_height_median" in position],
            dtype=np.float32,
        )
        x = float(np.median(xs))
        y = float(np.median(ys))
        distance = math.hypot(x, y)
        bearing = math.atan2(y, max(0.05, x))
        latest = positions[-1]
        captured = {
            "x": x,
            "y": y,
            "distance": distance,
            "bearing": bearing,
            "surface_height_median": float(np.median(surface_heights)) if len(surface_heights) > 0 else None,
            "surface_height_p20": latest.get("surface_height_p20"),
            "surface_height_p80": latest.get("surface_height_p80"),
            "mode": latest.get("mode", "unknown"),
            "samples": int(sum(position.get("samples", 0) for position in positions)),
            "position_samples": len(positions),
            "frame": latest.get("frame", ""),
            "fallback_detection": fallback_used,
        }

        if image_context:
            captured.update(image_context)
        return captured

    def classify_static_lying_surface(self, owner_candidate=None):
        if not self.lying_surface_classification_enabled:
            return "lying", "surface classification disabled", None

        position = self.capture_fallen_owner_position(owner_candidate)
        if position is None:
            return "lying", self.last_approach_failure_reason or "no point-cloud surface height", None

        surface_height = position.get("surface_height_median")
        bottom_norm = position.get("bbox_bottom_norm")
        center_y_norm = position.get("bbox_center_y_norm")
        ground_height_limit = self.lying_ground_max_surface_height
        furniture_height_limit = self.lying_furniture_min_surface_height

        if surface_height is not None and math.isfinite(float(surface_height)):
            if surface_height <= ground_height_limit:
                return (
                    "lying_ground",
                    "point-cloud surface height %.2fm <= ground limit %.2fm" % (surface_height, ground_height_limit),
                    position,
                )
            if surface_height >= furniture_height_limit:
                return (
                    "lying",
                    "point-cloud surface height %.2fm >= furniture limit %.2fm" % (surface_height, furniture_height_limit),
                    position,
                )

        image_ground_hint = (
            bottom_norm is not None
            and center_y_norm is not None
            and float(bottom_norm) >= self.lying_ground_bbox_bottom_ratio
            and float(center_y_norm) >= self.lying_ground_bbox_center_ratio
        )
        if image_ground_hint:
            return "lying_ground", "ambiguous height with low image bbox", position
        return "lying", "ambiguous/elevated lying posture", position

    def approach_fallen_owner(self, owner_candidate=None, initial_position=None):
        if not self.fall_approach_enabled:
            rospy.loginfo("Owner blind approach is disabled")
            return True

        position = initial_position or self.capture_fallen_owner_position(owner_candidate)
        if position is None:
            rospy.logwarn("Owner blind approach cannot start: %s", self.last_approach_failure_reason)
            return False

        standoff_distance = clamp(abs(self.fall_approach_standoff_distance), 0.45, 1.8)
        distance_tolerance = max(0.02, self.fall_approach_distance_tolerance)
        max_travel = max(0.0, self.fall_approach_max_travel_distance)
        travel_distance = clamp(position["distance"] - standoff_distance, 0.0, max_travel)
        lidar_guard_distance = max(0.30, self.fall_approach_lidar_stop_distance + self.fall_approach_lidar_margin)
        self.last_approach_failure_reason = "owner blind approach did not start"

        rospy.loginfo(
            "Owner blind approach snapshot: mode=%s distance=%.2f bearing=%.3f travel=%.2f standoff=%.2f lidar_guard=%.2f samples=%d/%d fallback_det=%s",
            position["mode"],
            position["distance"],
            position["bearing"],
            travel_distance,
            standoff_distance,
            lidar_guard_distance,
            position["position_samples"],
            position["samples"],
            position["fallback_detection"],
        )

        if travel_distance <= distance_tolerance:
            self.stop_base()
            rospy.loginfo("Owner is already within standoff distance")
            return True

        current_yaw = self.get_latest_yaw()
        if current_yaw is not None and abs(position["bearing"]) > self.approach_bearing_tolerance:
            target_yaw = current_yaw + position["bearing"]
            if not self.rotate_to_yaw(target_yaw, timeout=max(1.0, self.fall_approach_turn_timeout)):
                self.last_approach_failure_reason = "failed to align to owner before blind drive"
                rospy.logwarn("Owner blind approach aborted: %s", self.last_approach_failure_reason)
                return False
        elif current_yaw is None and abs(position["bearing"]) > self.approach_forward_bearing_limit:
            self.last_approach_failure_reason = "no odom yaw for large owner bearing: %.3f" % position["bearing"]
            rospy.logwarn("Owner blind approach aborted: %s", self.last_approach_failure_reason)
            return False

        start_xy = self.wait_for_odom_xy(timeout=1.0)
        start_time = time.time()
        last_loop_time = start_time
        timed_travelled = 0.0
        deadline = start_time + max(1.0, self.fall_approach_drive_timeout)
        rate = rospy.Rate(10)
        used_timed_fallback = start_xy is None
        if used_timed_fallback:
            rospy.logwarn("No odom position available; using timed owner blind drive")

        while not rospy.is_shutdown() and time.time() < deadline:
            front_distance = self.front_scan_distance()
            if front_distance is not None and front_distance <= lidar_guard_distance:
                self.stop_base()
                rospy.loginfo(
                    "Owner blind approach stopped at lidar guard: lidar=%.2f guard=%.2f",
                    front_distance,
                    lidar_guard_distance,
                )
                return True

            if start_xy is not None:
                travelled = self.xy_distance(start_xy, self.get_latest_odom_xy())
            else:
                travelled = timed_travelled
            if travelled is None:
                travelled = 0.0

            remaining = travel_distance - travelled
            if remaining <= distance_tolerance:
                self.stop_base()
                rospy.loginfo(
                    "Owner blind approach complete: travelled=%.2f target=%.2f timed_fallback=%s",
                    travelled,
                    travel_distance,
                    used_timed_fallback,
                )
                return True

            speed = clamp(
                self.fall_approach_linear_gain * remaining,
                self.fall_approach_min_linear_speed,
                self.fall_approach_linear_speed,
            )
            if front_distance is not None:
                clearance = front_distance - lidar_guard_distance
                speed = min(speed, max(0.0, clearance) * 0.35)

            twist = Twist()
            twist.linear.x = max(0.0, speed)
            self.cmd_pub.publish(twist)
            now = time.time()
            if start_xy is None:
                timed_travelled += twist.linear.x * max(0.0, now - last_loop_time)
            last_loop_time = now
            rospy.loginfo_throttle(
                1.0,
                "Owner blind approach: travelled=%.2f target=%.2f remaining=%.2f cmd=%.2f lidar=%.2f",
                travelled,
                travel_distance,
                remaining,
                twist.linear.x,
                front_distance if front_distance is not None else -1.0,
            )
            rate.sleep()

        self.stop_base()
        travelled = self.xy_distance(start_xy, self.get_latest_odom_xy()) if start_xy is not None else None
        if travelled is not None and travelled >= travel_distance - max(0.20, distance_tolerance):
            rospy.logwarn(
                "Owner blind approach timed out near target; accepting stop: travelled=%.2f target=%.2f",
                travelled,
                travel_distance,
            )
            return True
        self.last_approach_failure_reason = "owner blind drive timed out before target: travelled=%s target=%.2f" % (
            "%.2f" % travelled if travelled is not None else "unknown",
            travel_distance,
        )
        rospy.logwarn("Owner blind approach timed out: %s", self.last_approach_failure_reason)
        return False

    def advance_forward_by_distance(self, extra_distance, speed=None, timeout=None, lidar_guard_distance=None, label="owner"):
        target_distance = max(0.0, float(extra_distance))
        if target_distance <= 0.0:
            return True

        move_speed = abs(float(speed)) if speed is not None else self.fall_approach_extra_close_speed
        move_speed = max(0.03, move_speed)
        move_timeout = float(timeout) if timeout is not None else self.fall_approach_extra_close_timeout
        move_timeout = max(1.0, move_timeout)
        guard_distance = lidar_guard_distance
        if guard_distance is None:
            guard_distance = max(0.30, self.fall_approach_lidar_stop_distance + self.fall_approach_lidar_margin)

        start_xy = self.wait_for_odom_xy(timeout=1.0)
        start_time = time.time()
        last_loop_time = start_time
        timed_travelled = 0.0
        deadline = start_time + move_timeout
        rate = rospy.Rate(10)
        used_timed_fallback = start_xy is None

        rospy.loginfo(
            "%s extra-close advance: target=%.2fm speed=%.2fm/s guard=%.2fm timed_fallback=%s",
            label,
            target_distance,
            move_speed,
            guard_distance,
            used_timed_fallback,
        )

        while not rospy.is_shutdown() and time.time() < deadline:
            front_distance = self.front_scan_distance()
            if front_distance is not None and front_distance <= guard_distance:
                self.stop_base()
                rospy.loginfo(
                    "%s extra-close advance stopped by lidar guard: lidar=%.2f guard=%.2f",
                    label,
                    front_distance,
                    guard_distance,
                )
                return True

            if start_xy is not None:
                travelled = self.xy_distance(start_xy, self.get_latest_odom_xy())
            else:
                travelled = timed_travelled
            if travelled is None:
                travelled = 0.0

            remaining = target_distance - travelled
            if remaining <= 0.02:
                self.stop_base()
                rospy.loginfo(
                    "%s extra-close advance complete: travelled=%.2f target=%.2f",
                    label,
                    travelled,
                    target_distance,
                )
                return True

            twist = Twist()
            twist.linear.x = min(move_speed, max(0.0, remaining))
            if front_distance is not None:
                clearance = front_distance - guard_distance
                twist.linear.x = min(twist.linear.x, max(0.0, clearance) * 0.35)
            self.cmd_pub.publish(twist)
            now = time.time()
            if start_xy is None:
                timed_travelled += twist.linear.x * max(0.0, now - last_loop_time)
            last_loop_time = now
            rospy.loginfo_throttle(
                1.0,
                "%s extra-close advance: travelled=%.2f target=%.2f remaining=%.2f cmd=%.2f lidar=%.2f",
                label,
                travelled,
                target_distance,
                remaining,
                twist.linear.x,
                front_distance if front_distance is not None else -1.0,
            )
            rate.sleep()

        self.stop_base()
        travelled = self.xy_distance(start_xy, self.get_latest_odom_xy()) if start_xy is not None else None
        if travelled is not None and travelled >= target_distance - 0.10:
            rospy.logwarn(
                "%s extra-close advance timed out near target; accepting stop: travelled=%.2f target=%.2f",
                label,
                travelled,
                target_distance,
            )
            return True
        rospy.logwarn(
            "%s extra-close advance timed out before target: travelled=%s target=%.2f",
            label,
            "%.2f" % travelled if travelled is not None else "unknown",
            target_distance,
        )
        return False

    def approach_waving_owner(self, owner_candidate=None):
        if not self.approach_on_waving_enabled:
            rospy.loginfo("Approach after waving is disabled")
            return True

        if owner_candidate is not None and "det" in owner_candidate:
            det = owner_candidate["det"]
            image_width = float(owner_candidate.get("image_width", 0.0))
            if image_width > 0:
                self.owner_track_center = float(det.center_x) / image_width

        target_distance = clamp(abs(self.approach_standoff_distance), 0.35, 1.8)
        lidar_guard_distance = max(0.30, self.approach_lidar_stop_distance + self.approach_lidar_margin)
        rospy.loginfo(
            "Approaching waving owner with Kinect point cloud: target=%.2fm lidar_guard=%.2fm topic=%s",
            target_distance,
            lidar_guard_distance,
            self.points_topic,
        )
        deadline = time.time() + max(1.0, self.approach_timeout)
        stable_cycles = 0
        saw_position = False
        sent_forward_cmd = False
        last_position = None
        last_valid_time = 0.0
        last_cmd = Twist()
        stopped = True
        self.last_approach_failure_reason = "approach did not start"

        def publish_motion(cmd):
            nonlocal last_cmd, stopped
            alpha = clamp(self.approach_command_smoothing, 0.0, 1.0)
            smoothed = Twist()
            smoothed.linear.x = last_cmd.linear.x + (cmd.linear.x - last_cmd.linear.x) * alpha
            smoothed.angular.z = last_cmd.angular.z + (cmd.angular.z - last_cmd.angular.z) * alpha
            if abs(cmd.linear.x) < 1e-3:
                smoothed.linear.x = 0.0
            if abs(cmd.angular.z) < 1e-3:
                smoothed.angular.z = 0.0
            self.cmd_pub.publish(smoothed)
            last_cmd = smoothed
            stopped = False

        def stop_once():
            nonlocal last_cmd, stopped
            if not stopped:
                self.stop_base()
            last_cmd = Twist()
            stopped = True

        def coast_or_stop(reason):
            nonlocal stable_cycles
            stable_cycles = 0
            self.last_approach_failure_reason = reason
            if last_valid_time > 0.0 and time.time() - last_valid_time <= self.approach_missing_data_grace:
                coast = Twist()
                coast.linear.x = max(0.0, last_cmd.linear.x * clamp(self.approach_missing_linear_scale, 0.0, 1.0))
                coast.angular.z = last_cmd.angular.z
                publish_motion(coast)
                return
            stop_once()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            det, width, height = self.select_tracking_detection()
            front_distance = self.front_scan_distance()

            twist = Twist()
            if det is None or width is None or height is None or width <= 0 or height <= 0:
                coast_or_stop("lost owner detection during approach")
                rate.sleep()
                continue

            center_norm = float(det.center_x) / float(width)
            self.owner_track_center = center_norm
            position = self.estimate_owner_position_from_pointcloud(det, width, height)
            if position is None:
                coast_or_stop(self.last_pointcloud_reason or "no valid owner point cloud position")
                rate.sleep()
                continue

            saw_position = True
            last_position = position
            last_valid_time = time.time()
            distance = position["distance"]
            bearing = position["bearing"]
            distance_error = distance - target_distance
            lidar_blocked = front_distance is not None and front_distance <= lidar_guard_distance
            lidar_missing_required = self.approach_require_lidar and front_distance is None
            if lidar_missing_required:
                rospy.logwarn_throttle(2.0, "Waiting for valid front lidar before approaching owner")
                self.last_approach_failure_reason = "front lidar required but unavailable"
                stable_cycles = 0
                stop_once()
                rate.sleep()
                continue

            within_distance = distance <= target_distance + self.approach_distance_tolerance
            within_bearing = abs(bearing) <= self.approach_bearing_tolerance
            if lidar_blocked:
                stop_once()
                rospy.loginfo(
                    "Waving owner approach stopped at lidar guard: mode=%s pointcloud_dist=%.2f bearing=%.3f samples=%d lidar=%.2f",
                    position["mode"],
                    distance,
                    bearing,
                    position["samples"],
                    front_distance if front_distance is not None else -1.0,
                )
                return True

            if within_distance:
                stable_cycles += 1
                stop_once()
                if stable_cycles >= self.approach_arrival_stable_cycles:
                    rospy.loginfo(
                        "Waving owner point-cloud approach complete: mode=%s distance=%.2f bearing=%.3f bearing_ok=%s samples=%d lidar=%.2f",
                        position["mode"],
                        distance,
                        bearing,
                        within_bearing,
                        position["samples"],
                        front_distance if front_distance is not None else -1.0,
                    )
                    return True
                rate.sleep()
                continue

            stable_cycles = 0
            twist.angular.z = clamp(
                self.approach_angular_gain * bearing,
                -self.approach_max_angular_speed,
                self.approach_max_angular_speed,
            )
            forward_bearing_limit = max(self.approach_center_tolerance, self.approach_forward_bearing_limit)
            if abs(bearing) <= forward_bearing_limit and distance_error > self.approach_distance_tolerance:
                speed = clamp(
                    self.approach_linear_gain * distance_error,
                    self.approach_min_linear_speed,
                    self.approach_linear_speed,
                )
                if front_distance is not None:
                    clearance = front_distance - lidar_guard_distance
                    speed = min(speed, max(0.0, clearance) * 0.35)
                twist.linear.x = max(0.0, speed)
                sent_forward_cmd = sent_forward_cmd or twist.linear.x > 0.0
            elif distance_error > self.approach_distance_tolerance:
                self.last_approach_failure_reason = "owner bearing too large for forward motion: bearing=%.3f limit=%.3f distance=%.2f" % (
                    bearing,
                    forward_bearing_limit,
                    distance,
                )

            rospy.loginfo_throttle(
                1.0,
                "Owner point-cloud approach: mode=%s distance=%.2f err=%.2f bearing=%.3f cmd=(%.2f, %.2f) samples=%d lidar=%.2f",
                position["mode"],
                distance,
                distance_error,
                bearing,
                twist.linear.x,
                twist.angular.z,
                position["samples"],
                front_distance if front_distance is not None else -1.0,
            )
            publish_motion(twist)
            rate.sleep()

        stop_once()
        if last_position is not None and last_position["distance"] <= target_distance + max(0.20, self.approach_distance_tolerance):
            rospy.logwarn(
                "Waving owner approach timed out near target; accepting stop: distance=%.2f target=%.2f bearing=%.3f",
                last_position["distance"],
                target_distance,
                last_position["bearing"],
            )
            return True
        if saw_position:
            self.last_approach_failure_reason = (
                "timed out before reaching target; last distance=%.2f target=%.2f bearing=%.3f forward_cmd=%s"
                % (
                    last_position["distance"],
                    target_distance,
                    last_position["bearing"],
                    sent_forward_cmd,
                )
            )
        rospy.logwarn("Waving owner approach timed out: %s", self.last_approach_failure_reason)
        return False

    def run(self):
        self.set_yolo_paused(False)
        self.wait_for_hardware_inputs()

        if self.speak_on_start:
            self.say("我开始前往客厅寻找主人。")

        self.navigate_to_waypoint()

        if self.speak_on_arrival:
            self.say("我已到达客厅，开始寻找主人。")

        owner_candidate = self.scan_for_owner()
        if owner_candidate is None:
            self.say("我没有确认主人，请再给我一次机会。")
            return False

        if self.speak_on_owner_found:
            self.say("我已经识别到主人。")

        centered = self.center_owner_in_camera(owner_candidate)
        if not centered:
            if self.speak_on_finish:
                self.say("我已经识别到主人，但相机居中不稳定。")
            return True

        self.say("识别中。")
        action_label, action_confidence, action_reason = self.recognize_owner_action(owner_candidate)
        rospy.loginfo(
            "Owner action summary: label=%s confidence=%.2f reason=%s",
            action_label,
            action_confidence,
            action_reason,
        )
        normalized_action = str(action_label).strip().lower()
        initial_approach_position = None
        if normalized_action == "lying":
            normalized_action, lying_surface_reason, initial_approach_position = self.classify_static_lying_surface(
                owner_candidate
            )
            action_label = normalized_action
            rospy.loginfo(
                "Static lying surface verdict: label=%s reason=%s",
                normalized_action,
                lying_surface_reason,
            )

        self.say(self.action_to_speech(action_label), hold=self.action_speech_hold)
        if normalized_action in self.fall_approach_action_labels:
            approached = self.approach_fallen_owner(owner_candidate, initial_position=initial_approach_position)
            if approached:
                extra_close_completed = True
                if self.fall_approach_extra_close_enabled:
                    extra_close_completed = self.advance_forward_by_distance(
                        self.fall_approach_extra_close_distance,
                        speed=self.fall_approach_extra_close_speed,
                        timeout=self.fall_approach_extra_close_timeout,
                        label="owner",
                    )
                if normalized_action in self.fall_assist_arm_action_labels:
                    self.perform_fall_assist_arm_motion()
                elif normalized_action in ("lying", "sitting") and extra_close_completed:
                    self.wait_for_electrical_switch_instruction()
            else:
                rospy.logwarn("Owner blind approach did not complete for action=%s: %s", normalized_action, self.last_approach_failure_reason)
        elif normalized_action == "waving":
            approached = self.approach_waving_owner(owner_candidate)
            if approached:
                self.say(self.approach_help_prompt)
            else:
                reason = self.last_approach_failure_reason
                rospy.logwarn("Waving owner approach did not complete before help prompt: %s", reason)
                if "lidar guard" in reason:
                    self.say("我前方距离太近，先停在这里。" + self.approach_help_prompt)
                elif "point cloud" in reason or "ROI" in reason:
                    self.say("我看到您在挥手，但点云定位还不稳定，我先在这里听您指示。" + self.approach_help_prompt)
                else:
                    self.say("我看到您在挥手，正在这里听您指示。" + self.approach_help_prompt)
        return True


def main():
    rospy.init_node("task1_find_owner_real")
    node = None
    try:
        node = RealOwnerSearchBeforeAction()
        success = node.run()
        rospy.loginfo("task1_find_owner_real finished: success=%s", success)
    except MissingHardwareError as exc:
        rospy.logfatal("task1_find_owner_real failed: %s", exc)
        if node is not None:
            node.stop_base()
            try:
                node.say("没有收到机器人传感器数据，请检查Kinect、雷达和里程计。", hold=2.5)
            except Exception as say_exc:
                rospy.logwarn("Unable to speak hardware failure message: %s", say_exc)
    except Exception as exc:
        rospy.logfatal("task1_find_owner_real failed: %s", exc)
        if node is not None:
            node.stop_base()
        raise
    finally:
        if node is not None:
            node.stop_base()


if __name__ == "__main__":
    main()
