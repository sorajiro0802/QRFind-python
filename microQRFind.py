import numpy as np
import pyboof as pb
import cv2


def main():
    camera_id = 1
    window_name = "Micro QR Finder"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cap = cv2.VideoCapture(camera_id)
    # prepair microQR detector
    detector = pb.FactoryFiducial(np.uint8).microqr()
    
    while True:
        ret, image = cap.read()
        if ret:
            image_mono = cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2GRAY)
            # convert image class to match pyboof
            boof_img = pb.ndarray_to_boof(image_mono)
            detector.detect(boof_img)
            
            for mqr in detector.detections:
                info = mqr.message
                points = mqr.bounds.convert_tuple()
                points = [(int(i), int(j)) for i, j in points] # LeftUp, RightUp, RightDown, LeftDown <- int
                # line enclose mQRCode
                image = cv2.polylines(image, [np.array(points)], True, (255, 0, 0), 5)
        
            cv2.imshow(window_name, image)
        
        # when Esp pressed
        if cv2.waitKey(5)  & 0xFF==27:
            break
    
    cv2.destroyWindow(window_name)


if __name__=="__main__":
    main()