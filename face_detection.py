#Read a video stream from Camera(Frame By Frame)
import cv2

#Open the default webcam
cap = cv2.VideoCapture(0)

#to read the file or classifier which work on facial data
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:

	#return two values
	#ret gives boolean and frame gives video stream
	ret,frame = cap.read()

	#for gray frame
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret== False:
		continue

	#It has a special method detectMultiscale. CascadeClassifier has this method	
	#detectMultiScale(gray_frame,scaling_Factor,No of Neigbours)
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)	
	#this method only work on fixed size of image. Suppose our method is trained on image 100*100
	#So we shoudl change the size of image if its larger.So scaling factor help us.
	#scaling factor:- how much image size is reduced at each pass
	# if 1:3 then 30% reduction if 1:05 then each pass 5%
	#neighbours :- 	How many neighbours each candiate rectangle should have.
	#Higher value results in less detection but with higher quality. 3-6 is a good value




	#here the faces will contain the starting co-ordinate of faces and width and height
	#if multiple faces are there then it will return a tuple of x,y,w,h
	
	for (x,y,w,h) in faces:
		#Here this box or matrices or known as kernel will iterate through the image and see
		#it contain faces or not
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,139),2)



	#Display the frame	
	cv2.imshow("Face Detector",frame)
	#Display the gray frame
	#cv2.imshow("Gray Frame",gray_frame)
	
	#The wait key returns it 32 bit
	#So we take bitwise and convert it to 8 bit
	key_pressed = cv2.waitKey(1) &	0xFF
	if key_pressed == ord('q'):
		break;

cap.release()
cv2.destroyAllWindows()		
