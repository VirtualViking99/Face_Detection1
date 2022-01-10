print("running")
import cv2
from random import randrange

print("import complete")



#load pretrained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#classifier detects front facing faces
# training can take anywhere from 30 mins to multiple days


print('data loaded')

#choose a camera(web cam) to capture video off of
webcam = cv2.VideoCapture(0) #when you pass a string with a video filetype postfit as an argument, it will look for the filepath



print('Camera sellected')

print("Camera running")
#iterate over frames

while True:			# we want it to loop over the frames forever until the video ends or until we kill cam
	
	# read the current frame
	successful_frame_read, frame = webcam.read()	#returns a tuple; a bool between if it's reading, and the actual image
	
	#grayscale the frame
	#greywebcam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



	#Detect Faces
	face_coordinates = trained_face_data.detectMultiScale(frame)

	for (x,y,w,h) in face_coordinates:
		cv2.rectangle(frame, (x,y),(x+w,y+h), (0,255,0),2)


	#always pass in the latest variable
	cv2.imshow('Face detector', frame)
	key = cv2.waitKey(1)	

	###Stop if W key is pressed
	if key==81 or key==113:			# https://commons.wikimedia.org/wiki/File:ASCII-Table-wide.svg
		print("Camera has exited")
		break
	
# memory allocation and streams of data, clean up code using .release()
webcam.release()



#Detect Faces
#face_coordinates = trained_face_data.detectMultiScale(img_gray)

#print("Facial Coordinates: " + str(face_coordinates))
#returns upper left and lower right boundaries of the rectangle in the face


#dynamically superimpose a rectangle over the image face to display, WITH the color coordinates
'''
takes 
1. an image you want to draw a rectangle on
2. a tuple of the upper left handcoordinate and lower right coordinate, which was stored in face_coordinates
3. BGR Color coordinates
4. thickness of the border 
''' 

# for (x,y,w,h) in face_coordinates:						#because this is a list, we can loop over it
# 	cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),2)
   #cv2.rectangle(img, (x,y),(x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)),2)


#always pass in the latest variable

print("code complete")

