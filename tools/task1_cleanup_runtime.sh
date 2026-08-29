#!/usr/bin/env bash
set -u

source /opt/ros/noetic/setup.bash 2>/dev/null || true
source "$HOME/catkin_ws/devel/setup.bash" 2>/dev/null || true

echo "Stopping WPB task1 runtime nodes if ROS master is available..."
if rostopic list >/dev/null 2>&1; then
  rosnode kill \
    /task1_find_owner_real \
    /yoloworld \
    /yoloworld_debug_viewer \
    /kinect2_bridge \
    /kinect2_points_xyzrgb_qhd \
    /kinect2_points_xyzrgb_sd \
    /kinect2_points_xyzrgb_hd \
    /kinect2 \
    2>/dev/null || true
  rosnode cleanup >/dev/null 2>&1 || true
else
  echo "ROS master is not reachable; skipping rosnode cleanup."
fi

echo "Killing leftover user processes for task1 Kinect/YOLO nodes..."
pkill -u "$USER" -f 'task1_find_owner_real.py' 2>/dev/null || true
pkill -u "$USER" -f 'yoloworld_node.py' 2>/dev/null || true
pkill -u "$USER" -f 'yoloworld_debug_viewer.py' 2>/dev/null || true
pkill -u "$USER" -f 'nodelet.*kinect2' 2>/dev/null || true
pkill -u "$USER" -f 'kinect2_bridge' 2>/dev/null || true

sleep 2
echo "Remaining relevant processes:"
ps -eo pid,stat,etime,cmd | grep -E 'task1_find_owner|yoloworld|kinect2' | grep -v grep || true

echo "If Kinect remains at 0Hz after cleanup, unplug/replug the Kinect USB cable or reboot the robot PC."
