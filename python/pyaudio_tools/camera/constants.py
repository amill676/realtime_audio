__author__ = 'Adam Miller'

# Pan constants
MAX_PAN_DEGREE = 170.
MIN_PAN_DEGREE = -170.
MAX_PAN_HEX = -2448  # 0xF670
MIN_PAN_HEX = 2448  # 0x0990

# Tilt constants
MAX_TILT_DEGREE = 25
MIN_TILT_DEGREE = -90
TILT_CENTER_DEGREE = -42.5
MAX_TILT_HEX = -828  # 0xFCC4 corresponds to 25 degrees
MIN_TILT_HEX = 972  # 0x03CC corresponds to -90 degrees

# VISCA codes
ABSOLUTE_POS_STR = 81010602

# Movement speeds
MAX_PAN_SPEED = 0x18
MIN_PAN_SPEED = 0x00
MAX_TILT_SPEED = 0x14
MIN_TILT_SPEED = 0x00

# Command prepend str
VISCA_COMMAND_STR = '/command/ptzf.cgi?visca='
