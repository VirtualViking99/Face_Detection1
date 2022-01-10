print("running")
import cv2
from random import randrange

print("import complete")



#load pretrained 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#classifier detects front facing faces

print('data loaded')

#choose an image
img = cv2.imread('William Weidner, Manon Lescaut, Geronte 1.jpg')

print('image sellected')

#grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Image Grayed")



#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(img_gray)

print("Facial Coordinates: " + str(face_coordinates))
#returns upper left and lower right boundaries of the rectangle in the face


#dynamically superimpose a rectangle over the image face to display, WITH the color coordinates
'''
takes 
1. an image you want to draw a rectangle on
2. a tuple of the upper left handcoordinate and lower right coordinate, which was stored in face_coordinates
3. BGR Color coordinates
4. thickness of the border 
''' 

for (x,y,w,h) in face_coordinates:						#because this is a list, we can loop over it
	cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),2)
   #cv2.rectangle(img, (x,y),(x+w,y+h), (randrange(128,256),randrange(128,256),randrange(128,256)),2)




#always pass in the latest variable
cv2.imshow('Face detector', img)
print('image shown')




cv2.waitKey() #keeps window open

print("code complete")

