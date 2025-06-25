from ev3dev2.motor import MoveTank, OUTPUT_A, OUTPUT_B, OUTPUT_D, SpeedPercent, LargeMotor


tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
collector = LargeMotor(OUTPUT_D)

def move_robot(direction):
    normal_speed = 40  # default hastighed
    turn_speed = 7
    collector_speed = 70
    delivery_speed = -100
   

    if direction == "forward":
        print("Moving forward")
        tank_drive.on(SpeedPercent(normal_speed), SpeedPercent(normal_speed))
        collector.on(SpeedPercent(collector_speed))
 
    elif direction == "left":
        print("Turning left")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-turn_speed), SpeedPercent(turn_speed)) 

    elif direction == "right":
        print("Turning right")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(turn_speed), SpeedPercent(-turn_speed))

    elif direction == "backward":
        print("Reversing")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-20), SpeedPercent(-20))
    
    elif direction == "delivery":
        print("Delivering")
        collector.on(SpeedPercent(delivery_speed))
        tank_drive.on(SpeedPercent(0), SpeedPercent(0))
    
    elif direction == "continue":
        print("Continuing")

    elif direction == "very_slow_left":
        print("Turning very slowly left")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-0.5), SpeedPercent(0.5)) 

    elif direction == "very_slow_right":
        print("Turning very slowly right")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(0.5), SpeedPercent(-0.5))

    elif direction == "very_slow_forward":
        print("Moving very slowly forward")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(5), SpeedPercent(5)) 

    elif direction == "very_slow_backward":
        print("Reversing very slowly")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-4), SpeedPercent(-4))

    elif direction == "slow_left":
        print("Turning slowly left")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-1), SpeedPercent(1)) 

    elif direction == "slow_right":
        print("Turning slowly right")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(1), SpeedPercent(-1))

    elif direction == "slow_forward":
        print("Moving slowly forward")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(10), SpeedPercent(10)) 

    elif direction == "slow_backward":
        print("Reversing slowly")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-7), SpeedPercent(-7))

    elif direction == "medium_left":
        print("Turning medium left")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-3), SpeedPercent(3)) 

    elif direction == "medium_right":
        print("Turning medium right")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(3), SpeedPercent(-3))

    elif direction == "medium_forward":
        print("Moving medium forward")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(20), SpeedPercent(20)) 

    elif direction == "medium_backward":
        print("Reversing medium")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-10), SpeedPercent(-10))
    
    elif direction == "fast_left":
        print("Turning fast left")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-15), SpeedPercent(15)) 

    elif direction == "fast_right":
        print("Turning fast right")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(15), SpeedPercent(-15))

    elif direction == "fast_forward":
        print("Moving fast forward")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(80), SpeedPercent(80)) 

    elif direction == "fast_backward":
        print("Reversing fast")
        collector.on(SpeedPercent(collector_speed))
        tank_drive.on(SpeedPercent(-30), SpeedPercent(-30))

    elif direction == "stop":
        print("Stopping")
        tank_drive.off()
        collector.off()

    else:
        tank_drive.off() 
        collector.off()