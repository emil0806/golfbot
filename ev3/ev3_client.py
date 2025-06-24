import socket
from movement import move_robot, close_ass
from config import MAC_IP, PORT

# Connect to Mac
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((MAC_IP, PORT))

print("Connected to Mac")

current_command = None

start_count = 0

distance_timer = 0
dist_number = 0

try:
    while True:

        data = client_socket.recv(1024).decode().strip() 
        if not data:
            continue

        if data != current_command:
            move_robot(data)
            current_command = data 

except (socket.error, KeyboardInterrupt):
    move_robot("stop")
    client_socket.close()
