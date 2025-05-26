from ev3dev2.motor import MoveTank, OUTPUT_A, OUTPUT_B, OUTPUT_C, OUTPUT_D, SpeedPercent, LargeMotor, MediumMotor

tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
collector = LargeMotor(OUTPUT_D)
delivery = MediumMotor(OUTPUT_C)

collector.on(SpeedPercent(-35))


def move_robot(direction):
    collector.on(SpeedPercent(-35))

    if direction == "forward":
        print("Moving forward")
        tank_drive.on(SpeedPercent(35), SpeedPercent(35)) 

    elif direction == "left":
        print("Turning left")
        tank_drive.on(SpeedPercent(-15), SpeedPercent(15)) 

    elif direction == "right":
        print("Turning right")
        tank_drive.on(SpeedPercent(15), SpeedPercent(-15))

    elif direction == "backward":
        print("Reversing")
        tank_drive.on(SpeedPercent(-35), SpeedPercent(-35))
    
    elif direction == "delivery":
        print("Delivering")
        delivery.on_for_seconds(SpeedPercent(-5), 1)
    
    elif direction == "continue":
        print("Continuing")
        delivery.on_for_seconds(SpeedPercent(5), 1)

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