"""Coin Detection"""
import cv2
import numpy as np

def image():
    img = cv2.imread("Image\Total_Coin\T10_Coin (1).jpg")
    return img

def coin10():
    """Detection Coin10"""
    counter_coin10 = 0
    img_coin10 = image()
    img_coin10 = cv2.resize(img_coin10, (400, 400))
    cascade_coin10 = cv2.CascadeClassifier("C:\Project PSIT\cascade\Casecade_Coin1.xml")
    gray_img10 = cv2.cvtColor(img_coin10, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.03
    minNeighbors = 2
    coin10_detect = cascade_coin10.detectMultiScale(gray_img10, scaleFactor, minNeighbors)
    for (x, y, w, h) in coin10_detect:
        cv2.rectangle(img_coin10, (x, y), (x + w ,y + h), (0, 255, 0) ,thickness = 2)
        counter_coin10 += 1
    return counter_coin10

def coin5():
    """Detection Coin5"""
    counter_coin5 = 0
    img_coin5 = image()
    img_coin5 = cv2.resize(img_coin5, (400, 400))
    cascade_coin5 = cv2.CascadeClassifier("C:\Project PSIT\cascade\Casecade_Coin1.xml")
    gray_img5 = cv2.cvtColor(img_coin5, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.03
    minNeighbors = 2
    coin5_detect = cascade_coin5.detectMultiScale(gray_img5, scaleFactor, minNeighbors)
    for (x, y, w, h) in coin5_detect:
        cv2.rectangle(img_coin5, (x, y), (x + w ,y + h), (0, 255, 0) ,thickness = 2)
        counter_coin5 += 1
    return counter_coin5

def coin1():
    """Detection Coin1"""
    counter_coin1 = 0
    img_coin1 = image()
    img_coin1 = cv2.resize(img_coin1, (400, 400))
    cascade_coin1 = cv2.CascadeClassifier("C:\Project PSIT\cascade\Casecade_Coin1.xml")
    gray_img1 = cv2.cvtColor(img_coin1, cv2.COLOR_BGR2GRAY)
    scaleFactor = 1.03
    minNeighbors = 2
    coin1_detect = cascade_coin1.detectMultiScale(gray_img1, scaleFactor, minNeighbors)
    for (x, y, w, h) in coin1_detect:
        cv2.rectangle(img_coin1, (x, y), (x + w ,y + h), (0, 255, 0) ,thickness = 2)
        counter_coin1 += 1
    return counter_coin1

def CoinDetection():
    """CoinDetection"""
    cap = cv2.imread(image())
    while (cap.imread()):
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
            if area < 5000 or area > 35000:
                continue
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
            counter += 1
        E1 = cv2.FONT_HERSHEY_SIMPLEX
        E2 = cv2.LINE_AA
        cv2.putText(roi, str(counter), (10, 100), E1, 4, (255, 0, 0), 2, E2)
        cv2.imshow("Show", roi)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
CoinDetection()
