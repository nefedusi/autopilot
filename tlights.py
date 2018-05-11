from collections import deque
import numpy as np
import imutils
import cv2
import math

def filterLedOn(hsv):
    lowerLed = np.array([0,0,245])
    upperLed = np.array([180,180,255])
    return cv2.inRange(hsv, lowerLed, upperLed)

def filterBlack(hsv):
    lowerBlack1 = np.array([0, 0, 0])
    upperBlack1 = np.array([180, 255, 90])
    mask1 = cv2.inRange(hsv, lowerBlack1, upperBlack1)
    return mask1
    #lowerBlack2 = np.array([0, 0, 30])
    #upperBlack2 = np.array([180, 230, 160])
    #mask2 = cv2.inRange(hsv, lowerBlack2, upperBlack2)
    #return mask1 | mask2

def filterRedOn(hsv):
    lowerRed1 = np.array([0,80,130])
    upperRed1 = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    lowerRed2 = np.array([170,80,130])
    upperRed2 = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    return mask1 | mask2;

def filterYellowOn(hsv):
    lowerYellow = np.array([18,100,120])
    upperYellow = np.array([32,255,255])
    return cv2.inRange(hsv, lowerYellow, upperYellow)

def filterGreenOn(hsv):
    lowerGreen = np.array([33,100,110])
    upperGreen = np.array([80,255,255])
    return cv2.inRange(hsv, lowerGreen, upperGreen)

def findAndDrawCircles(imgOriginal, mask, circleColor=(155,0,0)):      # circleColor in BGR!!!
	circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=17,
                            param1=10, param2=11, minRadius=0, maxRadius=24)
	if (circles is not None):
		circles = np.uint16(np.around(circles))
		for j in circles[0,:]:
			cv2.circle(imgOriginal, (j[0],j[1]), j[2], circleColor, 2)
	return circles

def mouseCoords(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		if (frame is not None):
			bgr = frame[y,x]
			hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
			#print("click x=", x, "y=", y)
			print("hsv =", hsv)

def rotateCoords(x, y, teta): #teta angle must be in radians!
	xr =   x * math.cos(teta) + y * math.sin(teta)
	yr = - float(x * math.sin(teta)) + y * math.cos(teta) #if there wouldn't be float(...), there'd be wrong values 'cause of minus sign
	return xr, yr

def findSignal(circlesLed, circlesColor, rects, img):
	if ((circlesLed is not None) and (circlesColor is not None)):
		for i in circlesLed[0,:]:
			for j in circlesColor[0,:]:
				rl = i[2]
				rr = j[2]
				#dr = rl/rr
				if (rl < rr):
					bigR = rr
				else:
					bigR = rl
				dx = i[0] - j[0]
				dy = i[1] - j[1]
				if (math.sqrt(dx*dx + dy*dy) < 0.7 * bigR):
					for r in rects:
						rect = cv2.minAreaRect(r)
						box = np.int0(cv2.boxPoints(rect))
						w, h = rect[1][0], rect[1][1]
						if (h<w): h, w = w, h
						teta = math.radians(rect[2])
						xrmin, xrmax, yrmin, yrmax = float("inf"), -float("inf"), float("inf"), -float("inf") 
						for x, y in box:
							xr, yr = rotateCoords(x, y, teta)
							if (xr < xrmin): xrmin = xr
							if (xr > xrmax): xrmax = xr
							if (yr < yrmin): yrmin = yr
							if (yr > yrmax): yrmax = yr
						xp, yp = i[0], i[1]
						xpr, ypr = rotateCoords(xp, yp, teta)
						if (xpr > xrmin and xpr < xrmax and ypr > yrmin and ypr < yrmax):
							return True, True, int(h)
					return True, False, None
				#if ((0.5 < dr < 2) and ():
	return False, False, None

def detect(c):
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.09 * peri, True)
	if len(approx) == 4:
		rect = cv2.minAreaRect(approx)
		box = np.int0(cv2.boxPoints(rect))
		w = rect[1][0]
		h = rect[1][1]
		if (h<w): h, w = w, h
		ar = h / float(w)
		teta = math.radians(rect[2])
		if (ar >= 1.7 and ar <= 2.5):
			return approx	
	return None


sizeG = 7 	# size of Gauss filter kernel
camera = cv2.VideoCapture(0)
frame = None
cv2.namedWindow("Frame")
cv2.namedWindow("Mask")
cv2.setMouseCallback("Frame", mouseCoords)

while True:
	(grabbed, frame) = camera.read()
	frame = imutils.resize(frame, width=600) # numpy.ndarray
	blur = cv2.GaussianBlur(frame, (sizeG, sizeG), 0)

	hsvLedOn = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskLedOn = filterLedOn(hsvLedOn)
	maskLedOn = cv2.dilate(maskLedOn, None, iterations=2)
	maskLedOn = cv2.erode(maskLedOn, None, iterations=1)
	circlesLed = findAndDrawCircles(frame, maskLedOn, (155, 0, 0))

	hsvBlack = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskBlack = filterBlack(hsvBlack)
	maskBlack = cv2.dilate(maskBlack, None, iterations=2)
	maskBlack = cv2.erode(maskBlack, None, iterations=1)
	circlesBlack = findAndDrawCircles(frame, maskBlack, (155, 0, 0))
	
	hsvRedOn = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskRedOn = filterRedOn(hsvRedOn)
	maskRedOn = cv2.dilate(maskRedOn, None, iterations=2)
	maskRedOn = cv2.erode(maskRedOn, None, iterations=1)
	circlesRed = findAndDrawCircles(frame, maskRedOn, (0, 0, 155))

	hsvYellowOn = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskYellowOn= filterYellowOn(hsvYellowOn)
	maskYellowOn = cv2.dilate(maskYellowOn, None, iterations=2)
	maskYellowOn = cv2.erode(maskYellowOn, None, iterations=1)
	circlesYellow = findAndDrawCircles(frame, maskYellowOn, (0, 140, 200))

	hsvGreenOn = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskGreenOn = filterGreenOn(hsvGreenOn)
	maskGreenOn = cv2.dilate(maskGreenOn, None, iterations=2)
	maskGreenOn = cv2.erode(maskGreenOn, None, iterations=1)
	circlesGreen = findAndDrawCircles(frame, maskGreenOn, (0, 190, 0))

	lowThreshold = 25
	maskCanny = cv2.Canny(blur, lowThreshold, lowThreshold*3, apertureSize=3)
	cnts = cv2.findContours(maskBlack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	rects = []	
	for c in cnts:
		approx = detect(c)
		c = c.astype("int")
		cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
		if (approx is not None):
			rect = cv2.minAreaRect(approx)
			box = np.int0(cv2.boxPoints(rect))
			cv2.drawContours(frame, [box],0,(0,0,255),2)
			rects.append(approx)
	 	
	signal, inside, hboxr = findSignal(circlesLed, circlesRed, rects, frame)
	if (signal):
		text = "Red"
		if (inside): text += " in box"
		frame = cv2.putText(frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,200,255), 
			thickness=2, lineType=2)
			
	signal, inside, hboxy = findSignal(circlesLed, circlesYellow, rects, frame)
	if (signal):
		text = "Yellow"
		if (inside): text += " in box"
		frame = cv2.putText(frame, text, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,200,255), 
			thickness=2, lineType=2)
			
	signal, inside, hboxg = findSignal(circlesLed, circlesGreen, rects, frame)
	if (signal):
		text = "Green"
		if (inside): text += " in box"
		frame = cv2.putText(frame, text, (15, 90), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,200,255), 
			thickness=2, lineType=2)
			
	if (hboxg is not None): hbox = hboxg
	elif (hboxy is not None): hbox = hboxy
	else: hbox = hboxr
	frame = cv2.putText(frame, "Box height = " + str(hbox), (15, 115), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(0,200,255), 
			thickness=2, lineType=2)
	
	cv2.imshow("Frame", frame)
	#cv2.imshow("Mask", maskLedOn)
	cv2.imshow("Mask", maskBlack)
	#cv2.imshow("Mask", maskCanny)
	#cv2.imshow("Mask red on", maskRedOn)
	#cv2.imshow("Mask", maskYellowOn)
	#cv2.imshow("Mask green on", maskGreenOn)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
