from collections import deque
import numpy as np
import imutils
import cv2
import math, sys
import cv2.cv as cv
# camera = cv2.VideoCapture(0)
# fwidth = int(sys.argv[1])
# print("frame width =", fwidth)
global frame

def filterLed(hsv):
    lowerLed = np.array([0,0,240])
    upperLed = np.array([180,230,255])
    return cv2.inRange(hsv, lowerLed, upperLed)

def filterBlack1(hsv):
    lowerBlack1 = np.array([0, 0, 0])
    upperBlack1 = np.array([180, 255, 90])
    mask1 = cv2.inRange(hsv, lowerBlack1, upperBlack1)
    return mask1
    
def filterBlack2(hsv):
    lowerBlack2 = np.array([0, 0, 0])
    upperBlack2 = np.array([180, 230, 140])
    mask2 = cv2.inRange(hsv, lowerBlack2, upperBlack2)
    return mask2

def filterRed1(hsv):
    lowerRed1 = np.array([0,80,200])
    upperRed1 = np.array([10,255,255])
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    lowerRed2 = np.array([170,80,200])
    upperRed2 = np.array([180,255,255])
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    return mask1 | mask2;
    
def filterRed2(hsv):
    lowerRed1 = np.array([0,80,150])
    upperRed1 = np.array([10,255,200])
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    lowerRed2 = np.array([170,80,150])
    upperRed2 = np.array([180,255,200])
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    return mask1 | mask2;
    
def filterRed3(hsv):
    lowerRed1 = np.array([0,80,130])
    upperRed1 = np.array([10,255,150])
    mask1 = cv2.inRange(hsv, lowerRed1, upperRed1)
    lowerRed2 = np.array([170,80,130])
    upperRed2 = np.array([180,255,150])
    mask2 = cv2.inRange(hsv, lowerRed2, upperRed2)
    return mask1 | mask2;

def filterYellow(hsv):
    lowerYellow = np.array([18,100,120])
    upperYellow = np.array([32,255,255])
    return cv2.inRange(hsv, lowerYellow, upperYellow)

def filterGreen(hsv):
    lowerGreen = np.array([33,100,110])
    upperGreen = np.array([80,255,255])
    return cv2.inRange(hsv, lowerGreen, upperGreen)

def findAndDrawCirclesOnMask(frame, mask, circleColor=(155,0,0)):      # circleColor in BGR!!!
	circles = cv2.HoughCircles(mask, cv.CV_HOUGH_GRADIENT, dp=1, minDist=1, #minDist=int(fwidth*0.028),
                            param1=10, param2=10, minRadius=0, maxRadius=int(mask.shape[1]*0.06))
	if (circles is not None):
		circles = np.uint16(np.around(circles))
		#for j in circles[0,:]:
			#cv2.circle(frame, (j[0],j[1]), j[2], circleColor, 2)
	return circles

def findCircles(frame, blur, filterColor, color): #color is a tuple with 3 elements in BGR
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	mask = filterColor(hsv)
	mask = cv2.dilate(mask, None, iterations=2)
	mask = cv2.erode(mask, None, iterations=1)
	circles = findAndDrawCirclesOnMask(frame, mask, color)
	return mask, circles

def mouseCoords(event,x,y,flags,param):
	if event == cv2.EVENT_LBUTTONDOWN:
		if (frame is not None):
			bgr = frame[y,x]
			hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
			#print("click x=", x, "y=", y)
			# print("hsv =", hsv)

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
				if (rl < rr):
					bigR = rr
				else:
					bigR = rl
				dx = i[0] - j[0]
				dy = i[1] - j[1]
				if (math.sqrt(dx*dx + dy*dy) < 0.7 * bigR):
					"""for r in rects:
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
							return True, True, int(h)"""
					return True, False, None
	return False, False, None

def detectRect(c):
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.08 * peri, True)
	if len(approx) == 4:
		rect = cv2.minAreaRect(approx)
		w = rect[1][0]
		h = rect[1][1]
		if (h<w): h, w = w, h
		ar = h / float(w)
		teta = math.radians(rect[2])
		if (ar >= 1.6 and ar <= 2.5):
			return approx	
	return None

def detectSignal(circlesLed, circlesColor, rects, frame, colorName, y, glowingPeriod, 
	minGlowingDetectPeriod, maxGlowingPeriod=6):
	text = colorName
	glowingPeriodOld = glowingPeriod
	signal, inside, hbox = findSignal(circlesLed, circlesColor, rects, frame)
	if (signal):
		text += " now"
		if (inside):
			text += " in box"
			glowingPeriod += 2
		else:
			glowingPeriod += 1
		if (glowingPeriod > maxGlowingPeriod):
			glowingPeriod = maxGlowingPeriod
				
	if (not signal):
		glowingPeriod -= 1
		if (glowingPeriod < 0):
			glowingPeriod = 0
				
	if (glowingPeriod >= minGlowingDetectPeriod):
		text += ", detected state"
	#frame = cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,200,255), 
	#		thickness=2, lineType=2)
	#print text
	return hbox, glowingPeriod
	
def driveIsAllowed(frame, boxHeightThreshold, glowTime):
	glowingpr1, glowingpr2, glowingpr3, glowingpy, glowingpg = glowTime[:5]
	#print "cycle begin: ", glowingpr1, glowingpr2, glowingpr3, glowingpy, glowingpg
	sizeG = 7 	# size of Gauss filter kernel
	minGlowingDetectPeriod = 1
	blur = cv2.GaussianBlur(frame, (sizeG, sizeG), 0)

	
	hsvBlack1 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskBlack1 = filterBlack1(hsvBlack1)
	maskBlack1 = cv2.dilate(maskBlack1, None, iterations=2)
	maskBlack1 = cv2.erode(maskBlack1, None, iterations=1)

	hsvBlack2 = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
	maskBlack2 = filterBlack2(hsvBlack2)
	maskBlack2 = cv2.dilate(maskBlack2, None, iterations=2)
	maskBlack2 = cv2.erode(maskBlack2, None, iterations=1)
	
	maskRed1, circlesRed1 = findCircles(frame, blur, filterRed1, (0, 0, 200))
	maskRed2, circlesRed2 = findCircles(frame, blur, filterRed2, (0, 0, 150))
	maskRed3, circlesRed3 = findCircles(frame, blur, filterRed3, (0, 0, 100))
	maskYellow, circlesYellow = findCircles(frame, blur, filterYellow, (0, 150, 150))
	maskGreen, circlesGreen = findCircles(frame, blur, filterGreen, (0, 200, 0))
	maskLed, circlesLed = findCircles(frame, blur, filterLed, (155, 0, 0))

	rects = []	
	"""cnts = cv2.findContours(maskBlack1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	cnts2 = cv2.findContours(maskBlack2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts2 = cnts2[0] if imutils.is_cv2() else cnts2[1]
	cnts += cnts2

	
	for c in cnts:
		approx = detectRect(c)
		c = c.astype("int")
		#cv2.drawContours(frame, [c], -1, (0, 255, 0), 1)
		if (approx is not None):
			rect = cv2.minAreaRect(approx)
			#box = np.int0(cv2.boxPoints(rect))
			#cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
			rects.append(approx)"""
	 	
	hboxr1, glowingpr1 = detectSignal(circlesLed, circlesRed1, rects, frame, "Red1", 30, glowingpr1, 
		minGlowingDetectPeriod)
	hboxr2, glowingpr2 = detectSignal(circlesLed, circlesRed2, rects, frame, "Red2", 50, glowingpr2, 
		minGlowingDetectPeriod)
	hboxr3, glowingpr3 = detectSignal(circlesLed, circlesRed3, rects, frame, "Red3", 70, glowingpr3, 
		minGlowingDetectPeriod)
	glowingpr = max((glowingpr1, glowingpr2, glowingpr3))
	hboxy, glowingpy = detectSignal(circlesLed, circlesYellow, rects, frame, "Yellow", 90, glowingpy, 
		minGlowingDetectPeriod)
	hboxg, glowingpg = detectSignal(circlesLed, circlesGreen, rects, frame, "Green", 110, glowingpg, 
		minGlowingDetectPeriod)
	print "glowing periods red: ", glowingpr1, glowingpr2, glowingpr3
	print "glowing periods: ", glowingpr, glowingpy, glowingpg
			
	"""if (hboxg is not None): hbox = hboxg
	elif (hboxy is not None): hbox = hboxy
	elif (hboxr1 is not None): hbox = hboxr1
	elif (hboxr2 is not None): hbox = hboxr2
	else: hbox = hboxr3
	print "box height =", hbox
	#frame = cv2.putText(frame, "Box height = " + str(hbox), (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 
	#	fontScale=0.6, color=(0,200,255), thickness=2, lineType=2)"""
		
	#cv2.imshow("led", maskLed)
	#cv2.imshow("Maskb1", maskBlack1)
	#cv2.imshow("Maskb2", maskBlack2)
	#cv2.imshow("Maskro1", maskRed1)
	#cv2.imshow("Maskro2", maskRed2)
	#cv2.imshow("Maskro3", maskRed3)
	#cv2.imshow("Mask", maskYellowOn)
	#cv2.imshow("Mask green on", maskGreenOn)	
	
	#if (glowingpr >= minGlowingDetectPeriod and hbox > boxHeightThreshold):
	glowTime[:5] = glowingpr1, glowingpr2, glowingpr3, glowingpy, glowingpg
	if (glowingpr >= minGlowingDetectPeriod):
		return False
	return True



# frame = None
#cv2.namedWindow("Maskb1")
#cv2.namedWindow("Maskb2")
#cv2.namedWindow("Maskro1")
#cv2.namedWindow("Maskro2")
#cv2.namedWindow("Maskro3")
# cv2.namedWindow("Frame")
# cv2.setMouseCallback("Frame", mouseCoords)


# while True:
# 	(grabbed, frame) = camera.read()
# 	frame = imutils.resize(frame, fwidth) # numpy.ndarray

# 	print("driveIsAllowed =", driveIsAllowed(frame, fwidth*0.25))
	
# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF
# 	if key == ord("q"):
# 		break
	
# # cleanup the camera and close any open windows
# camera.release()
# cv2.destroyAllWindows()
