import cv2
import argparse
import os,sys
import shutil
#INITIALIZE THE MAIN DATABASE PATH
mainpath =r'C:\Users\Niyati\Desktop\FaceRecog\Phase-01'

#Arguments needed to get the Id of student
ap = argparse.ArgumentParser()
ap.add_argument("-i","--ID",required=True,help="The identification number of student")
args=vars(ap.parse_args())

#Get the paths
training_dir = os.path.join(mainpath,'training_dir')
print(training_dir)
if(os.path.exists(training_dir)):
	pass
else:
	os.mkdir(training_dir)
#training_dir = os.listdir('./training_dir/')
person_dir=os.path.join(training_dir,args["ID"])
if(os.path.exists(person_dir)):
	o=int(input("The student database already exists .Modify existing database? 1:Yes 0:No"))
	if o==1:
		"""print(person_dir)
		shutil.rmtree(person_dir)
		os.mkdir(person_dir)"""
		for fil in os.listdir(person_dir):
			os.remove(os.path.join(person_dir,fil))
		print("Overwriting the database")
	elif o==0:
		exit
else:
	os.mkdir(person_dir)

#Capture images 
cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0
"""
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
"""
while True:
       
       if img_counter<5:
              
              k= cv2.waitKey(1)
              ret,frame = cam.read()
              cv2.imshow("Capture",frame)
              if k%256 == 32:
                     
                     fileName = os.path.join(person_dir,"IMAGE "+str(img_counter)+".png")
                     cv2.imwrite(fileName,frame)
                     img_counter+=1
                     print("Image captured")
       else:

              print("5 images Captured")
              break
			  
cam.release()

cv2.destroyAllWindows()