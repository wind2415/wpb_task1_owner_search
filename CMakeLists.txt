cmake_minimum_required(VERSION 3.0.2)
project(wpb_task1_owner_search)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  cv_bridge
  geometry_msgs
  move_base_msgs
  nav_msgs
  perception_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
)

catkin_package(
  CATKIN_DEPENDS actionlib actionlib_msgs cv_bridge geometry_msgs move_base_msgs nav_msgs perception_msgs rospy sensor_msgs std_msgs std_srvs tf
)

catkin_install_python(PROGRAMS
  scripts/robot_mic_test.py
  scripts/task1_find_owner_real.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config data docs launch maps tools
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
