import numpy as np
import pyboof as pb
import cv2


def main():
    camera_id = 0
    window_name = "Micro QR Finder"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(camera_id)
    # prepair microQR detector
    detector = pb.FactoryFiducial(np.uint8).microqr()
    
    while True:
        ret, image = cap.read()
        if ret:
            image_mono = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
            boof_img = pb.ndarray_to_boof(image_mono)
            detector.detect(boof_img)
            
            for mqr in detector.detections:
                print(mqr.message)
                print(getDiagonalPoints(mqr.bounds.convert_tuple()))
        
            cv2.imshow(window_name, image)
            
        if cv2.waitKey(5)  & 0xFF==ord('q'):
            break
    
    cv2.destroyWindow(window_name)

def getDiagonalPoints(corner):
    # points4 have 4 points of corners of mQRcode.
    # this function returns points of LeftUp and RightDown
    return (corner[0], corner[2])

if __name__=="__main__":
    main()