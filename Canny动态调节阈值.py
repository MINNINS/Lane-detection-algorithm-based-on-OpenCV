import cv2

cv2.namedWindow('edge_detection')
cv2.createTrackbar('minThreshold','edge_detection',50,1000,lambda x:x)
cv2.createTrackbar('maxThreshold','edge_detection',100,1000,lambda x:x)

capture=cv2.VideoCapture('vedion_01.flv')
ret,img=capture.read()
img=cv2.cvtColor(img,cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('train02.png',cv2.IMREAD_GRAYSCALE)

while True:
    minThreshold=cv2.getTrackbarPos('minThreshold','edge_detection')
    maxThreshold=cv2.getTrackbarPos('maxThreshold','edge_detection')
    edges=cv2.Canny(img,minThreshold,maxThreshold)
    cv2.imshow('edge_detection',edges)
    cv2.waitKey(10)