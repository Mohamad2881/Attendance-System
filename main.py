import numpy as np
import face_recognition
from urllib.request import *

from utlis import *


path = './reference images'
images, imgNames = get_images(path)

# markAttendace('Khalifa')
knownEncodesList = findEncodings(images)
print("Encoding Complete")

# You may need to change this Url
url = 'http://192.168.1.3:8080/shot.jpg'
while True:
    # read the image from the mobile app
    imgResp = urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    img = cv2.flip(img, 1)
    # Resize image and change it to RGB
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect Faces and find their encodings
    currframefaces = face_recognition.face_locations(imgS)
    currframeEncode = face_recognition.face_encodings(imgS, currframefaces)

    # Compare the detected faces with the known ones
    for faceloc, face_encoding in zip(currframefaces, currframeEncode):
        matches = face_recognition.compare_faces(knownEncodesList, face_encoding)
        facedis = face_recognition.face_distance(knownEncodesList, face_encoding)
        best_match_index = np.argmin(facedis)

        if matches[best_match_index]:
            name = imgNames[best_match_index].upper()
            # print(name)
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0))
            cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

            # save the name and time in csv file
            markAttendace(name)

    # Show the image
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

