import cv2
import detect

img, imgxyxy = detect.detect('all/images/box9.jpg')

cv2.imshow("detect, crop",img)
print(img.shape)
cv2.waitKey()

