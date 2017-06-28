# This message describes a FingerHand object as in gpg/src/finger_hand.h

std_msgs/Int64 forward_axis # the index of the horizontal axis in the hand frame (grasp approach direction), usually 0
std_msgs/Int64 lateral_axis # the index of the vertical axis in the hand frame (closing direction of the robot hand), usually 1

std_msgs/Float64 init_bite # the initial bit of the hand, whatever it is that means
std_msgs/Float64 hand_height # the height of the hand

std_msgs/Float64 finger_width # the width of the robot hand fingers
std_msgs/Float64 hand_depth # the hand depth (finger length)
std_msgs/Float64 hand_outer_diameter # the hand depth (finger length)

std_msgs/Float64 bottom # the base of the hand
std_msgs/Float64 top # the top of the hand, where the fingertips are
std_msgs/Float64 left # the left side of the gripper bounding box
std_msgs/Float64 right # the right side of the gripper bounding box
std_msgs/Float64 center # the horizontal center of the gripper bounding box
std_msgs/Float64 surface # the corresponding vertical base point of the hand in the point cloud