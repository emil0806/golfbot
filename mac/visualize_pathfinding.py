import matplotlib.pyplot as plt
import math

def visualize_pathfinding(back_marker, ball_positions):
    (rx, ry), (fx, fy), _ = back_marker

    # If one ball is passed, convert to list
    if isinstance(ball_positions[0], (int, float)):
        ball_positions = [ball_positions]

    # Determine best ball
    best_ball = min(ball_positions, key=lambda ball: math.hypot(ball[0] - rx, ball[1] - ry))

    # Vectors
    vector_front = (fx - rx, fy - ry)
    vector_to_best = (best_ball[0] - rx, best_ball[1] - ry)

    # Plot setup
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_title("Robot Direction vs All Balls")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Robot center & front
    ax.plot(rx, ry, 'ko', label='Robot Center')
    ax.plot(fx, fy, 'bo', label='Robot Front')

    # Plot all balls
    for i, (bx, by) in enumerate(ball_positions):
        color = 'green' if (bx, by) != best_ball else 'red'
        marker = 'o' if (bx, by) != best_ball else 'X'
        label = f'Ball {i}' if (bx, by) != best_ball else f'Best Ball ({i})'
        ax.plot(bx, by, marker=marker, color=color, label=label)

    ax.annotate("", xy=(fx, fy), xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", lw=2, color="blue"))
    ax.plot([], [], color='blue', label='Facing Vector')  # dummy for legend

    # Plot vector to best ball
    ax.annotate("", xy=best_ball, xytext=(rx, ry),
                arrowprops=dict(arrowstyle="->", lw=2, color="red"))
    ax.plot([], [], color='red', label='To Best Ball')  # dummy for legend


    # Angle between vectors
    dot = vector_front[0] * vector_to_best[0] + vector_front[1] * vector_to_best[1]
    mag_f = math.hypot(*vector_front)
    mag_b = math.hypot(*vector_to_best)
    angle = 0
    if mag_f > 0 and mag_b > 0:
        cos_theta = max(-1, min(1, dot / (mag_f * mag_b)))
        angle = math.degrees(math.acos(cos_theta))

    ax.text(rx - 50, ry - 50, f"Angle: {angle:.2f}°", fontsize=10)

    ax.legend()
    plt.show()
