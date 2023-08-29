cmake_minimum_required(VERSION 3.0.2)
project(panda_description)

find_package(catkin REQUIRED COMPONENTS
  roslaunch
)


catkin_package(
)


include_directories(
  ${catkin_INCLUDE_DIRS}
)
