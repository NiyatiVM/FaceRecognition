import pickle
import face_recognition
from sklearn import svm
import os,sys
from sklearn.model_selection import train_test_split
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-i","--id",required=True,help="ID number of student")
args=vars(ap.parse_args())


mainpath=r'C:\Users\Niyati\Desktop\FaceRecog\Phase-01'
directory='train_dir'
dir_path = os.path.join(mainpath,directory)
person_dir=os.path.join(dir_path,args["id"])

filepath=r'C:\Users\Niyati\Desktop\FaceRecog\Phase-01\encodings\encodings.pickle'
with open(filepath,'rb') as f:
    data = pickle.load(f)
print(data)
encodings=data["encodings"]
names=data["names"]

pix = os.listdir(person_dir)
for person_img in pix:
	face=face_recognition.load_image_file(os.path.join(person_dir,person_img))
	face_bounding_boxes = face_recognition.face_locations(face)
	if len(face_bounding_boxes)==1:
		face_enc = face_recognition.face_encodings(face)[0]
		encodings.append(face_enc)
		names.append(args["id"])
	else:
		print(args["id"] + "/" + person_img + " was skipped and can't be used for training because number of faces found is "+str(len(face_bounding_boxes)))
print("[INFO] serializing encodings...")

#print(names)
#print(data)
ndata={"encodings":encodings,"names":names}
print(ndata)
encoding_dir=os.path.join(mainpath,"encodings")
if(os.path.exists(encoding_dir)):
	pass
else:
	os.mkdir(encoding_dir)
print("Done")
#f=open(os.path.join(encoding_dir,"e"+str(person)+".pickle"),"a")
f=open(os.path.join(encoding_dir,"encodings.pickle"),"wb")
pickle.dump(ndata,f)
f.close
