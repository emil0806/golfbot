import socket
from movement import move_robot
from config import MAC_IP, PORT

# Connect to Mac
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((MAC_IP, PORT))

print("Connected to Mac")

current_command = None

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
