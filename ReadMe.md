# Golfbot - How to Run the System

## 1. Transfer code to the EV3 robot

Open a terminal on your computer and connect to the robot via SSH:
```sh
ssh robot@172.20.10.2
```
Enter the password when prompted.

In a **new terminal window/tab** (on your computer, in the golfbot folder), transfer the EV3 code to the robot:
```sh
scp -r ev3/ robot@172.20.10.2:/home/robot/
```

## 2. Start the Mac server

In the same terminal (on your computer), change to the `mac` folder and start the server:
```sh
cd mac
python3 mac_server.py
```

## 3. Start the EV3 client

Go back to the terminal where you are connected to the robot via SSH. Change to the `ev3` folder and start the client:
```sh
cd ev3
python3 ev3_client.py
```

## 4. Stop the robot

Press `Ctrl+C` in both terminals to stop the programs.

---

**Summary of commands:**

On your computer:
```sh
ssh robot@172.20.10.2
scp -r ev3/ robot@172.20.10.2:/home/robot/
cd mac
python3 mac_server.py
```

On the robot (after SSH login):
```sh
cd ev3
python3 ev3_client.py
```






