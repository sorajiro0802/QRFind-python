# Author : Sora Kojima
# From : Tokyo Denki University
# Date : 2023/03/22

import cv2
import numpy as np

# initial
camera_id = 1  # usually 0 indicate facial camera of PC
window_name = "OpenCV QR Code"
font = cv2.FONT_HERSHEY_SIMPLEX
qcd = cv2.QRCodeDetector()
video = cv2.VideoCapture(camera_id)

while True:
    tick = cv2.getTickCount()
    ret, frame = video.read()
    if ret:
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
        if ret_qr:
            for i, j in zip(decoded_info, points):
                if i:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                frame = cv2.polylines(frame, [j.astype(int)], True, color, 8)
                frame = cv2.putText(img=frame,
                    text=str(i),
                    org=([int(j[0][k]-10) for k in range(len(j[0]))]),
                    fontFace=font,
                    fontScale=0.5,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_AA)
        retval, points = qcd.detect(frame)
        # if retval:
        #     print(points)
        # display FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
        cv2.putText(frame, f"{np.floor(fps)}fps", (60, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
        elapsed_time = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
        cv2.imshow(window_name, frame)
    # when ESC button pressed, the window break
    if cv2.waitKey(5) & 0xFF==27:
        break

cv2.destroyWindow(window_name)