import cv2

#default variable
counter = 0
#Read image
img = cv2.imread("C:\Project coin\Test.png")
img = cv2.resize(img, (1200, 800))

#Read classification(file cascade_coin5)
coin5_cascade = cv2.CascadeClassifier("C:\Project coin\cascadetest.xml")

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#classify coins1
scaleFactor = 1.15
mincoin = 2
coin_detect = coin5_cascade.detectMultiScale(gray_img, scaleFactor, mincoin)

#classify coins2
scaleFactor1 = 1.20
mincoin1 = 2
coin_detect1 = coin5_cascade.detectMultiScale(gray_img, scaleFactor1, mincoin1)

#Show at the coin5 find location1
for (x, y, w, h) in coin_detect:
    cv2.rectangle(img, (x, y), (x + w ,y + h), (0, 255, 0) ,thickness = 2)
    counter += 1

#Show at the coin5 find location1
for (x, y, w, h) in coin_detect1:
    cv2.rectangle(img, (x, y), (x + w ,y + h), (255, 0, 0) ,thickness = 2)
    counter += 1

#Show result
cv2.putText(img, str(counter), (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 200, 0), 2, cv2.LINE_AA)
cv2.imshow("Original",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
