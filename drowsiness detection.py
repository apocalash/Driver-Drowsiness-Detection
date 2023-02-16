import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# to sound the alarm if eyes are shut
mixer.init()
sound = mixer.Sound('alarm.wav')

# inputting the haar files that have the stores models and computation
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl = ['Close', 'Open']

# importing the saved model
model = load_model('Models_new/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(1)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [[99,99]]
lpred = [[99,99]]

# reading the live video feed from opencv and setting up the roi as left and right eye
while (True):  # infinite loop as its a video feed and continuous checking is needed
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # converting to grayscale cause it is easier to read it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detecting the face  and storing it to face variable
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    # passing grey image to detect eye
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # mapping the roi frame around the face
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # for each eye, detecting the eye and building the frame
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)  # converting from color to grey
        r_eye = cv2.resize(r_eye, (24, 24))  # rezise image to 24x24 matrix as trained, as as model to train
        r_eye = r_eye / 255  # noramalization to get 0 or 1 values
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = (model.predict(r_eye) > 0.5).astype("int32")  # predicting right value and convertitng to int , true of false value is coverted to 0 or 1
        if (rpred[0][1] == 1):  # binary classificationq
            lbl = 'Open'
        if (rpred[0][1] == 0):
            lbl = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = (model.predict(l_eye) > 0.5).astype("int32")

        if (lpred[0][0] == 1):
            lbl = 'Open'
        if (lpred[0][0] == 0):
            lbl = 'Closed'
        break


    if (rpred[0][1] == 0 and lpred[0][1] == 0):
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # checking the score value to detect if the eyes are shut or it was blink, if sleepy -> the alarm is sounded

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 5):
        # person is feeling sleepy so we beep the alarm
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False #alarm is turned off is score is less than 5
            pass
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # to exit from the infinte loop
        break
cap.release()
cv2.destroyAllWindows()
