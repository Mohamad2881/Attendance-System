import cv2
import face_recognition
import os
from datetime import datetime

def get_images(path):
    images = []
    imgNames = []

    # List image names in 'path' folder (e.g ____.jpg)
    mylist = os.listdir(path)

    for img in mylist:
        currimg = cv2.imread(f'{path}/{img}')
        images.append(currimg)

        # getting img name without (.jpg)
        imgNames.append(os.path.splitext(img)[0])
    return images, imgNames


# Find Face encoding for every image
def findEncodings(images):
    encodelist = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encodelist.append(encode)
    return encodelist


# This function takes in a name and save it with the current time if
# it is not already existed in the csv
def markAttendace(name):
    with open('attendance.csv', 'r+') as f:
        datalist = f.readlines()
        print(datalist)
        namelist = []
        for line in datalist:
            entry = line.split(',')
            namelist.append(entry[0])
        print(namelist)
        if name not in namelist:
            print("Asd")
            currtime = datetime.now()
            timestr = currtime.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{timestr}')
