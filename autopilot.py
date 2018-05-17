import cv2
import numpy as np
import math
import sys
import serial
import road
import cv2
import time
# print sys.argv[1]
# frame = cv2.imread(sys.argv[1])
capture = cv2.VideoCapture(0)
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=0)
def getCommand(engine, speed):
    command = engine
    if speed >= 0:
        command += "+"
    else:
        command += "-"
    abs_speed = abs(speed)
    str_speed = str(abs_speed)
    if abs_speed < 100:
        str_speed = "0" + str_speed
    if abs_speed < 10:
        str_speed = "0" + str_speed
    return command + str_speed

def sendSignal(l_speed, r_speed):
    command = getCommand('R', -int(l_speed))
    command += getCommand('L', -int(r_speed))
    ser.write(command)

speed = 90
rotate = 60
fcount = 0
log_file = open("log.txt", "w")
while True:
    start = time.time()
    grabbed, frame = capture.read()
# frame = cv2.imread("simple_line.png")
# frame = cv2.imread("curve_left_line.png")
#        if fcount  < 10:
#           fcount += 1
#          cv2.waitKey(1)
#         continue

    # sh = frame.shape
    sh = frame.shape
    # average_point =
    # print average_point, average_point_center
    frame = cv2.resize(frame, (sh[1]/4, sh[0]/4))
    #cv2.imshow("Autopilot", frame)
   # cv2.waitKey(1)
   # continue   
    weightened_image, angle = road.find_road_and_get_angle(frame)
    if angle is None or abs(angle) > math.pi/2:
        #if fcount > 1:
        l_speed = 0
        r_speed = 0
        fcount += 1
    else:
        fcount = 0
        l_speed = speed - rotate*angle
        r_speed = speed + rotate*angle
    print l_speed, r_speed
    log_file.write(str(l_speed) + " " + str(r_speed) + "\n")
    # angle =
    sendSignal(l_speed, r_speed)
    # sendSignal('R', r_speed)
    if weightened_image is None:
        weightened_image = frame
    # cv2.imwrite(sys.argv[1].split(".")[0] + "mod.jpg", weightened_image)
# while True:
    cv2.imshow("Autopilot", weightened_image)
    cv2.waitKey(10)
    end = time.time()
    print end - start
