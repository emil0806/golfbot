from ev3dev2.motor import MoveTank, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent, LargeMotor, MediumMotor

tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
collector = LargeMotor(OUTPUT_D)
delivery = MediumMotor(OUTPUT_C)

collector.on(SpeedPercent(50))


def move_robot(direction):
    speed = 60  # default hastighed
    turn_speed = 7
    collector_speed = 50

    collector.on(SpeedPercent(collector_speed))

    if direction == "forward":
        print("Moving forward")
        tank_drive.on(SpeedPercent(speed), SpeedPercent(speed)) 

    elif direction == "left":
        print("Turning left")
        tank_drive.on(SpeedPercent(-turn_speed), SpeedPercent(turn_speed)) 

    elif direction == "right":
        print("Turning right")
        tank_drive.on(SpeedPercent(turn_speed), SpeedPercent(-turn_speed))

    elif direction == "backward":
        print("Reversing")
        tank_drive.on(SpeedPercent(-20), SpeedPercent(-20))
    
    elif direction == "delivery":
        print("Delivering")
        tank_drive.on(SpeedPercent(0), SpeedPercent(0))
        delivery.on_for_seconds(SpeedPercent(-5), 2)
    
    elif direction == "continue":
        print("Continuing")
        delivery.on_for_seconds(SpeedPercent(5), 2)

    elif direction == "slow_left":
        print("Turning slowly left")
        tank_drive.on(SpeedPercent(-2), SpeedPercent(2)) 

    elif direction == "slow_right":
        print("Turning slowly right")
        tank_drive.on(SpeedPercent(3), SpeedPercent(-3))

    elif direction == "slow_forward":
        print("Moving slowly forward")
        tank_drive.on(SpeedPercent(20), SpeedPercent(20)) 

    elif direction == "slow_backward":
            print("Reversing slowly")
            tank_drive.on(SpeedPercent(-10), SpeedPercent(-10))
    
    elif direction == "fast_left":
        print("Turning fast left")
        tank_drive.on(SpeedPercent(-15), SpeedPercent(15)) 

    elif direction == "fast_right":
        print("Turning fast right")
        tank_drive.on(SpeedPercent(15), SpeedPercent(-15))

    elif direction == "fast_forward":
        print("Moving fast forward")
        tank_drive.on(SpeedPercent(80), SpeedPercent(80)) 

    elif direction == "fast_backward":
            print("Reversing fast")
            tank_drive.on(SpeedPercent(-30), SpeedPercent(-30))

    elif direction == "stop":
        print("Stopping")
        tank_drive.off()
        collector.off()

    #elif direction ==  "quit":
    #    tank_drive.off() 
    #    collector.off()

    else:
        tank_drive.off() 
        collector.off()