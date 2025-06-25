from enum import Enum

class RobotState(Enum):
    COLLECTION = 1
    DELIVERY = 2
    CORNER = 3
    COMPLETE = 4
