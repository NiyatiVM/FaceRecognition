#Import all important modules
import pickle
import face_recognition
from sklearn import svm
import os,sys
from sklearn.model_selection import train_test_split

#Defining paths
mainpath =r'C:\Users\Niyati\Desktop\FaceRecog\Phase-01'#Initialize the path 
training_dir = os.listdir(os.path.join(mainpath,'train_dir'))
train_dir = os.path.join(mainpath,'train_dir')
#encodings and names
encodings = []
names = []
#Training all images
#Loop through each training image for the current person
#Get the face encodings for the face in each image file
#If training image contains exactly one face
# Add face encoding for current image with corresponding label (name) to the training data
for person in training_dir:
	person_dir = os.path.join(train_dir,person)
	pix = os.listdir(person_dir)
	for person_img in pix:
		face=face_recognition.load_image_file(os.path.join(person_dir,person_img))
		face_bounding_boxes = face_recognition.face_locations(face)
		if len(face_bounding_boxes)==1:
			face_enc = face_recognition.face_encodings(face)[0]
			encodings.append(face_enc)
			names.append(person)
		else:
			print(person + "/" + person_img + " was skipped and can't be used for training because number of faces found is "+str(len(face_bounding_boxes)))
	print("[INFO] serializing encodings...")
	data={"encodings":encodings,"names":names}
	#print(data)
	encoding_dir=os.path.join(mainpath,"encodings")
	if(os.path.exists(encoding_dir)):
		pass
	else:
		os.mkdir(encoding_dir)
	#f=open(os.path.join(encoding_dir,"e"+str(person)+".pickle"),"wb")
f=open(os.path.join(encoding_dir,"encodings.pickle"),"wb")
pickle.dump(data,f)
f.close

"""
test_image = face_recognition.load_image_file('test_image.jpg')
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)
"""
