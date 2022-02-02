import cv2
import numpy as np

"""
    鼠标点击，获取当前坐标值
"""

img = cv2.imread("/media/lorenzo/26DA3A93DA3A5F6D/data/datasets/SemanticSegmentation/Irish_labels/p1_r75.tif")


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d, %d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image", 0)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.resizeWindow("image", 2000, 2000)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(100)
    except Exception:
        cv2.destroyWindow("image")
        break

cv2.waitKey(0)
cv2.destroyAllWindow()