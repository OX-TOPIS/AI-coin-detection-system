"""Coin Detection"""
import cv2

def maininput():
    """Videocap"""
    cap = cv2.VideoCapture("C:\Project PSIT\Video\Total_Coin\Total_Coin (2).mp4") #video
    return cap

def coin10():
    """Detection Coin10"""
    cap = maininput()
    face_cascade = cv2.CascadeClassifier("C:\Project PSIT\cascade\cascade_Coin_5 (200).xml") #xml
    counter_coin10 = 0
    while cap.isOpened():
        check, frame = cap.read()
        if check == True:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detect = face_cascade.detectMultiScale(gray_image, 1.2, 5)
            for (x, y, w, h) in face_detect:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                cv2.imshow('Output', frame)
                counter_coin10 += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    print(counter_coin10)
coin10()
