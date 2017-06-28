# This message describes a FingerHand object as in gpg/src/finger_hand.h

int32 forward_axis # the index of the horizontal axis in the hand frame (grasp approach direction), usually 0
int32 lateral_axis # the index of the vertical axis in the hand frame (closing direction of the robot hand), usually 1

float64 init_bite # the initial bit of the hand, whatever it is that means
float64 hand_height # the height of the hand

float64 finger_width # the width of the robot hand fingers
float64 hand_depth # the hand depth (finger length)
float64 hand_outer_diameter # the hand depth (finger length)

float64 bottom # the base of the hand
float64 top # the top of the hand, where the fingertips are
float64 left # the left side of the gripper bounding box
float64 right # the right side of the gripper bounding box
float64 center # the horizontal center of the gripper bounding box
float64 surface # the corresponding vertical base point of the hand in the point cloud