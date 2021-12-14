"""Coin Detection"""
import cv2
import numpy as np

def maininput():
    """Videocap"""
    cap = cv2.VideoCapture("Video\Total_Coin\Total_Cion (2).mp4")
    return cap

def coin10():
    """Detection Coin10"""
    cap = maininput()
    face_cascade = cv2.CascadeClassifier("cascade\Casecade_Coin10.xml")
    counter_coin10 = 0
    while cap.isOpened():
        check, frame = cap.read()
        if check == True:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detect = face_cascade.detectMultiScale(gray_image, 1.2, 5)
            for (x, y, w, h) in face_detect:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                counter_coin10 += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    return counter_coin10

def coin5():
    """Detection Coin5"""
    cap = maininput()
    face_cascade = cv2.CascadeClassifier("cascade\Casecade_Coin5.xml")
    counter_coin5 = 0
    while cap.isOpened():
        check, frame = cap.read()
        if check == True:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detect = face_cascade.detectMultiScale(gray_image, 1.2, 5)
            for (x, y, w, h) in face_detect:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                counter_coin5 += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    return counter_coin5

def coin1():
    """Detection Coin1"""
    cap = maininput()
    face_cascade = cv2.CascadeClassifier("cascade\Casecade_Coin1.xml")
    counter_coin1 = 0
    while cap.isOpened():
        check, frame = cap.read()
        if check == True:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_detect = face_cascade.detectMultiScale(gray_image, 1.2, 5)
            for (x, y, w, h) in face_detect:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                counter_coin1 += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    return counter_coin1

def CoinDetection():
    """CoinDetection"""
    cap = maininput()
    while (cap.read()):
        ref, frame = cap.read()
        ref = ref
        roi = frame[:1080, 0:1920]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        C1 = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        C2 = cv2.THRESH_BINARY_INV
        thresh = cv2.adaptiveThreshold(gray_blur, 255, C1, C2, 11, 1)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 4)
        result_img = closing.copy()
        D1 = cv2.RETR_EXTERNAL
        D2 = cv2.CHAIN_APPROX_SIMPLE
        contours, hierachy = cv2.findContours(result_img, D1, D2)
        hierachy = hierachy
        counter = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4000 or area > 28000:
                continue
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
            counter += 1
        E1 = cv2.FONT_HERSHEY_SIMPLEX
        E2 = cv2.LINE_AA
        cv2.putText(roi, str(coin1()) + (coin5()*5) + (coin10()*10), (10, 100), E1, 4, (255, 0, 0), 2, E2)
        cv2.imshow("Show", roi)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
CoinDetection()
