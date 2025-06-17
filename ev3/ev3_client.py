import socket
from movement import move_robot, close_ass
from config import MAC_IP, PORT
from ev3dev2.sensor.lego import UltrasonicSensor
from ev3dev2.sensor import INPUT_4
import time

us = UltrasonicSensor(INPUT_4)

# Connect to Mac
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((MAC_IP, PORT))

print("Connected to Mac")

current_command = None

start_count = 0

distance_timer = 0
dist_number = 0

try:
    if(start_count == 0):
        close_ass()
        start_count = 1
    while True:

        data = client_socket.recv(1024).decode().strip() 
        if not data:
            continue

        if data != current_command:
            move_robot(data)
            current_command = data 

        current_time = time.time()
        dist_number = 0
        if(current_time - distance_timer > 2):
            distance = us.distance_centimeters
            try:
                dist_number = str(distance)
                client_socket.sendall(dist_number.encode())
                distance_timer = current_time
            except socket.error:
                break

except (socket.error, KeyboardInterrupt):
    move_robot("stop")
    client_socket.close()
