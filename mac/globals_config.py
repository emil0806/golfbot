import math
import numpy as np

##### BORDER #####

FIELD_X_MIN = None
FIELD_X_MAX = None
FIELD_Y_MIN = None
FIELD_Y_MAX = None

PIX_FIELD_CORNERS = [
    (261, 50),      # TL
    (1588, 53),     # TR
    (1603, 1027),   # BR
    (259, 1030)     # BL
]

FIELD_LINES = []

def set_field_lines_from_corners():

    tl, tr, br, bl = PIX_FIELD_CORNERS
    lines = [
        (tl[0], tl[1], tr[0], tr[1]),  # top
        (tr[0], tr[1], br[0], br[1]),  # right
        (br[0], br[1], bl[0], bl[1]),  # bottom
        (bl[0], bl[1], tl[0], tl[1])   # left
    ]

    global FIELD_LINES
    FIELD_LINES = lines

def get_field_lines():
    return FIELD_LINES

def set_field_bounds_by_corners():

    xs = [pt[0] for pt in PIX_FIELD_CORNERS]
    ys = [pt[1] for pt in PIX_FIELD_CORNERS]

    global FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX
    FIELD_X_MIN = min(xs)
    FIELD_X_MAX = max(xs)
    FIELD_Y_MIN = min(ys)
    FIELD_Y_MAX = max(ys)


def get_field_bounds():
    return FIELD_X_MIN, FIELD_X_MAX, FIELD_Y_MIN, FIELD_Y_MAX

##### CROSS #####
CROSS_X_MIN = None
CROSS_X_MAX = None
CROSS_Y_MIN = None
CROSS_Y_MAX = None
CROSS_CENTER = None

CROSS_LINES = []

def set_cross_lines(lines):
    global CROSS_LINES
    CROSS_LINES = lines

def get_cross_lines():
    return CROSS_LINES

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


def extract_cross_lines(detected_lines):
    if not detected_lines:
        set_cross_lines([])
        return

    horizontal_lines = []
    vertical_lines = []

    for x1, y1, x2, y2 in detected_lines:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > abs(dy):  # Horisontale linjer
            horizontal_lines.append((x1, y1, x2, y2))
        else:  # Vertikale linjer
            vertical_lines.append((x1, y1, x2, y2))

    horizontal_fit, m_h, c_h = fit_line_to_points(horizontal_lines, vertical=False)
    vertical_fit, m_v, c_v = fit_line_to_points(vertical_lines, vertical=True)

    cross_lines = []
    if horizontal_fit:
        cross_lines.append(horizontal_fit)
    if vertical_fit:
        cross_lines.append(vertical_fit)

    set_cross_lines(cross_lines)
    print(f"c: {cross_lines}")

    if m_h is not None and m_v is not None:   
        try:
            y_intersect = int((m_h * c_v + c_h) / (1 - m_h * m_v))
            x_intersect = int(m_v * y_intersect + c_v)
        except ZeroDivisionError:
            x_intersect = y_intersect = 0  # fallback hvis linjerne er næsten parallelle

        x_min = min(horizontal_fit[0], horizontal_fit[2], vertical_fit[0], vertical_fit[2])
        x_max = max(horizontal_fit[0], horizontal_fit[2], vertical_fit[0], vertical_fit[2])
        y_min = min(horizontal_fit[1], horizontal_fit[3], vertical_fit[1], vertical_fit[3])
        y_max = max(horizontal_fit[1], horizontal_fit[3], vertical_fit[1], vertical_fit[3])

        set_cross_bounds(
            {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
            (x_intersect, y_intersect)
        )

def fit_line_to_points(lines, vertical=False):
    if not lines:
        return None, None, None

    points = []
    for x1, y1, x2, y2 in lines:
        points.append((x1, y1))
        points.append((x2, y2))

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    if len(xs) < 2:
        return None, None, None

    if vertical:
        # Fit x = m*y + c
        A = np.vstack([ys, np.ones(len(ys))]).T
        m, c = np.linalg.lstsq(A, xs, rcond=None)[0]

        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        x_min = int(m * y_min + c)
        x_max = int(m * y_max + c)

        return (x_min, y_min, x_max, y_max), m, c
    else:
        # Fit y = m*x + c
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, c = np.linalg.lstsq(A, ys, rcond=None)[0]

        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(m * x_min + c)
        y_max = int(m * x_max + c)

        return (x_min, y_min, x_max, y_max), m, c
