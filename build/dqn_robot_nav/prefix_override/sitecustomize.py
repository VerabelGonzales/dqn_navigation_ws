import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/verabel/dqn_navigation_ws/install/dqn_robot_nav'
