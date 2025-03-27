from ev3dev2.motor import MoveTank, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent, LargeMotor

tank_drive = MoveTank(OUTPUT_A, OUTPUT_B)
collector = LargeMotor(OUTPUT_C)


def move_robot(direction):
    collector.on(SpeedPercent(-100))

    if direction == "forward":
        print("Moving forward")
        tank_drive.on(SpeedPercent(30), SpeedPercent(30)) 

    elif direction == "left":
        print("Turning left")
        tank_drive.on(SpeedPercent(-20), SpeedPercent(20)) 

    elif direction == "right":
        print("Turning right")
        tank_drive.on(SpeedPercent(20), SpeedPercent(-20)) 

    elif direction == "stop":
        print("Stopping")
        tank_drive.off() 
        collector.off()

    else:
        tank_drive.off() 
        collector.off()