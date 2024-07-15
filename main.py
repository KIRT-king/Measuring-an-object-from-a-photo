import cv2
import numpy as np

import utlis

webcam = False
path = "images/10.jpg"
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 720) #width
cap.set(4, 480) #height

scale = 3
wP = 210 * scale
hP = 297 * scale

while True:
    if webcam:
        suc, img = cap.read()
    else:
        img = cv2.imread(path)

    imgContors, conts = utlis.getCintours(img, showCanny=False, minArea=50000, filter=4, draw = False)

    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utlis.warpImg(img, biggest, wP, hP)

        imgContors2, conts2 = utlis.getCintours(imgWarp, showCanny=False, minArea=2000,
                                                         filter=4, draw = False,
                                                         cThr=[50, 50], thk=3)
        if len(conts2) != 0:
            for obj in conts2:
                #cv2.polylines(imgContors2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utlis.reorder(obj[2])
                nW = round((utlis.FindDis(nPoints[0][0]//  scale, nPoints[1][0] // scale)/10),1)
                nH = round((utlis.FindDis(nPoints[0][0] // scale, nPoints[2][0] // scale)/10),1)
                cv2.arrowedLine(imgContors2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContors2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContors2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContors2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow("TOTAL:", imgContors2)


    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Original", img)
    cv2.waitKey(1)