# Author : Sora Kojima
# From : Tokyo Denki University
# Date : 2023/03/22

import cv2
import numpy as np
import matplotlib.pyplot as plt

# initial
camera_id = 0  # usually 0 indicate facial camera of PC
window_name = "OpenCV QR Code"
font = cv2.FONT_HERSHEY_SIMPLEX
qcd = cv2.QRCodeDetector()
video = cv2.VideoCapture(camera_id)

# # -----------------  main処理-------------------------
# # 1.retval : QRコードの検出。True or False
# # 2.decoded_info : QRコードに格納された文字列のタプル。検出できてもデコードできなかった場合は空文字となる (tuble)
# # 3.points : 検出できたQRコードの四隅の座標を表す (ndarray)
# # 4.straight_qrcode : QRコードそのものの2次元画像(ndarray)
# retval, decoded_info, points, straight_qrcode = qcd.detectAndDecodeMulti(img)

# # showing straight_qrcode test
# # plt.imshow(cv2.cvtColor(straight_qrcode[1], cv2.COLOR_BGR2RGB))
# # plt.show()
# # ------------------------------------------
# img = cv2.polylines(img, points.astype(int), True, (0, 255, 0), 3)
# for i, j in zip(decoded_info, points):
#     img = cv2.putText(img, i, j[0].astype(int),
#                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

while True:
    tick = cv2.getTickCount()
    ret, frame = video.read()
    if ret:
        ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
        if ret_qr:
            for i, j in zip(decoded_info, points):
                if i:
                    print(i)
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                frame = cv2.polylines(frame, [j.astype(int)], True, color, 8)
        retval, points = qcd.detect(frame)
        if retval:
            print(points)
        # display FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)
        cv2.putText(frame, f"{np.floor(fps)}fps", (60, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
        elapsed_time = (cv2.getTickCount() - tick) / cv2.getTickFrequency()
        cv2.imshow(window_name, frame)
    # when ESC button pressed, the window break
    if cv2.waitKey(5) & 0xFF==27:
        break

cv2.destroyWindow(window_name)