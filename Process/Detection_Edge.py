import cv2
import numpy as np
def CoinDetection():
    """CoinDetection"""
    cap = cv2.VideoCapture("C:\Project PSIT\Video\Total_Coin\Total_Coin (1).mp4")
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
            if area < 2900 or area > 28000:
                continue
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
            counter += 1
        E1 = cv2.FONT_HERSHEY_SIMPLEX
        E2 = cv2.LINE_AA
        #cv2.putText(roi, str(counter), (10, 100), E1, 4, (255, 0, 0), 2, E2)
        cv2.imshow("Show", gray_blur)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
CoinDetection()
