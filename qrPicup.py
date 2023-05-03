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
    dataDirPath = getFolderAbsPath(sys.argv)
    saveDirName = "output"
    saveDirdPath = f"{os.path.curdir}/{saveDirName}"
    
    # make save Directory
    if not os.path.exists(saveDirdPath):
        os.makedirs(saveDirdPath)

    img_path = "./data/mQRArray_rotated.png" #* tmp img
    # select QR or microQR
    decorder = pb.FactoryFiducial(np.uint8).microqr()
    img_name = os.path.basename(img_path)
    num, res = decodeQR(decorder, img_path)
    convertInfo2CSV(img_name, num, res, saveDirdPath, csv_name)
    
    
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
        raise ValueError("invalid commandLine argumant length")

def decodeQR(decorder, img_path) -> dict:
    image = pb.load_single_band(img_path, np.uint8)
    decorder.detect(image)
    num = len(decorder.detections)
    res = {}
    for code in decorder.detections:
        info = code.message
        points = code.bounds.convert_tuple()
        # convert float num of points to int
        points = [(int(i), int(j)) for i, j in points]
        res[info] = points
    # return dict ex.( {info : [ponts]} )
    return num, res
    
def convertInfo2CSV(img_name, num, dict_, save_dir, save_name):
    # format : <imgFileName>, <number of detected QRCode>, (x0,y0),(x1,y1),(x2,y2),(x3,y3), <decode string>, ...
    save_path = f"{save_dir}/{save_name}"
    with open(save_path, "w") as f:
        writer = csv.writer(f)
        writeLine = []
        writeLine.append(img_name)  # imgFileName
        writeLine.append(num)       # number of detected QRCodes
        # prepare csv form of QR infomations
        for info, points in dict_.items():
            writeLine.append(info)
            for x, y in points:
                writeLine.append(x)
                writeLine.append(y)
        writer.writerow(writeLine)


if __name__=="__main__":
    main()