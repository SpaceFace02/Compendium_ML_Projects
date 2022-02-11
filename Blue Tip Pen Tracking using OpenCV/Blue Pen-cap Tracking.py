import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture(0)

pts = deque(maxlen=64)

Lower_green = np.array([110,50,50])  # IN HSV
Upper_green = np.array([130,255,255])

while True:
	ret , frame = cap.read()
	hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2HSV)
	kernel = np.ones((5,5))
	mask = cv2.inRange(hsv , Lower_green ,Upper_green)
	# To erode the mask so that a thinner line is formed.
	mask = cv2.erode(mask , kernel = kernel , iterations = 2)
	# Opening just erodes then dilates removing noise outside.
	mask = cv2.morphologyEx(mask , cv2.MORPH_OPEN ,kernel)
	# Closing just dilates and erodes , removes noise inside the main roi
	mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE , kernel)
	mask = cv2.dilate(mask , kernel ,iterations = 1)
	res  = cv2.bitwise_and(frame ,frame ,mask = mask)

	contours , heirarchy = cv2.findContours(mask.copy() ,cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)[-2:]

	if len(contours)> 0:
		c = max(contours ,key = cv2.contourArea)
		(x,y) ,radius = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		centroid = (int(M['m10'] / M['m00']) , int(M['m01'] / M['m00']))

		if radius > 5:
			# Draw a circle around the pen cap.
			cv2.circle(frame ,(int(x) , int(y)) , int(radius) , (0,0,255) , -1)
			cv2.circle(frame, centroid, 5, (0, 0, 255), -1)

	# As the pen cap keeps on moving , the centroid keeps on changing , so we append the new points everytime.
		pts.appendleft(centroid)

	for i in range(1 ,len(pts)):
		# As float increases , the thickness of line decreases.
		thickness = int(np.sqrt(len(pts) / float(i+1))*2.5)
		# Line is drawn from one point to another point.
		cv2.line(frame , pts[i-1] , pts[i] ,(0,255,0) ,thickness)

	cv2.imshow('Frame' , frame)
	cv2.imshow('mask' , mask)
	cv2.imshow('res' , res)

   # Duration after which frame is displayed.
	if cv2.waitKey(1) == 27:
		break
		cap.release()
		cv2.destroyAllWindows()
		

# https://stackoverflow.com/questions/10022018/how-to-debug-indentation-errors-in-python#:~:text=You%20start%20using%20a%20text,enforce%20that%20in%20your%20editor.&text=I%20encountered%20a%20similar%20problem,%22Convert%20Indentation%20to%20Tabs%22.




