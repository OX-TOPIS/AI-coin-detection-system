import cv2
import numpy as np

cap=cv2.VideoCapture("video1.mp4")

while(cap.read()):
    ref, frame = cap.read() #ดึงค่าแต่ละตัวมาทำงาน
    roi = frame[0:1080, 0:1920] #จองพื้นที่ 1080x1920
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #ขาวดำ
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0) #สี0-255 ต้องเอามาเปลี่ยนเป็นขาว-ดำ
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1) #11 is boxsize #1 is + ค่าคงที่+-0
    kernel = np.ones((3, 3), np.uint8) #(1, 1) =ขนาด
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations = 3) #iterations=1ทำซ้ำ1รอบ

    result_img = closing.copy()
    contours, hiirachy = cv2.findContours(result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for countour in contours:
        area = cv2.contourArea(countour)
        if area < 5000:
            continue
        ellipse = cv2.fitEllipse(countour)
        cv2.ellipse(roi, ellipse, (0, 255, 0), 2)
    cv2.imshow("Show", roi) #Show
    roi = frame[:1080, 0:1920]
    if cv2.waitKey(1) & 0xFF == ord("q"): #เพื่อที่จะหยุดทำงาน
        break
cap.releasa()
cv2.destroyAllWindows()
