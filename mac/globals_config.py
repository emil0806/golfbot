##### BORDER #####
FIELD_X_MIN = 303
FIELD_X_MAX = 1615
FIELD_Y_MIN = 55
FIELD_Y_MAX = 1021

def set_field_bounds(bounds):
    global FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX
    FIELD_X_MIN = bounds["x_min"]
    FIELD_X_MAX = bounds["x_max"]
    FIELD_Y_MIN = bounds["y_min"]
    FIELD_Y_MAX = bounds["y_max"]

def get_field_bounds():
    return (FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX)

##### CROSS #####
CROSS_X_MIN = None
CROSS_X_MAX = None
CROSS_Y_MIN = None
CROSS_Y_MAX = None
CROSS_CENTER = None

def set_cross_bounds(bounds, center):
    global CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX, CROSS_CENTER
    CROSS_X_MIN = bounds["x_min"]
    CROSS_X_MAX = bounds["x_max"]
    CROSS_Y_MIN = bounds["y_min"]
    CROSS_Y_MAX = bounds["y_max"]
    CROSS_CENTER = center

def get_cross_bounds():
    return (CROSS_X_MIN, CROSS_X_MAX, CROSS_Y_MIN, CROSS_Y_MAX)

def get_cross_center():
    return CROSS_CENTER
