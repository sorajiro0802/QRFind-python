import sys
import os
import csv

import numpy as np
import pyboof as pb
import cv2

def main():
    #* feat add parser
    dataFoldPath = getFolderAbsPath(sys.argv)
    print(dataFoldPath)
    saveFoldName = "output"
    saveFoldPath = f"{os.path.curdir}/{saveFoldName}"
    
    img_path = "./data/mQRArray_rotated.png" #* tmp img
    # select QR or microQR
    decorder = pb.FactoryFiducial(np.uint8).microqr()
    res = decodeQR(decorder, img_path)
    
    
def getFolderAbsPath(cmd_arg):
    # check command line arguments
    if len(cmd_arg) == 2:
        # check 
        if len(cmd_arg[1].split(".")) != 2:
            absFolderPath = os.path.abspath(os.path.curdir) + "/" +cmd_arg[1]
            return absFolderPath
        else:
            raise NotADirectoryError
    else:
        raise ValueError("too many commandLine argumants")

def decodeQR(decorder, img_path) -> dict:
    image = pb.load_single_band(img_path, np.uint8)
    decorder.detect(image)
    res = {}
    for code in decorder.detections:
        info = code.message
        points = code.bounds.convert_tuple()
        # convert float num of points to int
        points = [(int(i), int(j)) for i, j in points]
        res[info] = points
    
    # return dict ex.( {info : [ponts]} )
    return res
    
def convertDict2CSV(res, save_path):
    pass


if __name__=="__main__":
    main()