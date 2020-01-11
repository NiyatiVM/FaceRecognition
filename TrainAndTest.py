import pickle
import face_recognition
import matplotlib.pyplot as plt
from sklearn import svm
import os,sys
from sklearn.model_selection import train_test_split,GridSearchCV

#For training and testing we divide the data
filepath=r'C:\Users\Niyati\Desktop\FaceRecog\Phase-01\encodings\encodings.pickle'
with open(filepath,'rb') as f:
    data = pickle.load(f)
encodings=data["encodings"]
names=data["names"]
X_train, X_test, y_train, y_test = train_test_split(encodings, names, test_size = 0.2, random_state=0)
clf = svm.SVC(gamma='scale')
clf.fit(X_train,y_train)
clf_predictions = clf.predict(X_test)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100 ))

#Testing the accuracy
test_image = face_recognition.load_image_file('test_image.jpg')
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(*name)