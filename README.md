# FaceRecognition
This project is built for a classroom monitoring system .
You can change the mainpath at the top of each file with the path of your base directory 
First , we capture images using capture.py ,which shows boundary box of faces , and stores the image after applying CLAHE .
from cmd 'python capture.py -i IDNUMBER'--give your id number in place of IDNUMBER
Next we encode all the images using Encoding.py and from cmd 'python Encoding.py'
Next we train with svm and test for accuracy as 'python TrainAndTest.py'
if new students are added , we do not need to create all encodings again , so use IndividualEncoding.py as 'python IndividualEncoding.py -i IDNUMBER'
Instead of capturing the data , we can also utilize other images from internet using prepare_dataset.py as '   python prepare_dataset.py -q 'QUERYNAME'   'you can use any person's name whose database you want to create in place of QUERYNAME , you'll need to get the API_key for it.
