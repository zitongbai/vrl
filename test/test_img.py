import cv2
import numpy as np


img = np.random.randint(0, 255, (120, 160), dtype=np.uint8)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)
