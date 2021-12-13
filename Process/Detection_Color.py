import cv2
import numpy as np

while True:
    img = cv2.imread("10-2.jpg")
    img = cv2.resize(img, (400, 400))
    lower = np.array([20, 72, 80])
    upper = np.array([105, 209, 234])
    mask = cv2. inRange(img, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("Ori", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", result)
    if cv2.waitKey(0) & 0xFF == ord("e"):
        break
    #cv2.imshow("Result", result)
cv2.destroyAllWindows()
