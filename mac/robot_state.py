from enum import Enum, auto

class RobotState(Enum):
    COLLECTION = 1
    DELIVERY = 2
    CORNER = 3
    COMPLETE = 4
