import cv2
import numpy as np
import time


# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('people.mp4')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0
while True:
    ret, frame = cam.read()
    # if not ret:
    #     print("Không thể nhận diện camera!")
    #     break

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    time.sleep(0.3)
    for (x, y, w, h) in faces:
        roi = cv2.resize(frame[y+2: y+h-2, x+2: x+w-2], (100, 100))
        cv2.imwrite('imgs_roi/roi_{}.jpg'.format(count), roi)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 255, 50), 2)
        count += 1

    cv2.imshow("Face Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()