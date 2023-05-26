import sys
import os
import csv
import datetime
from glob import glob

import numpy as np
import pyboof as pb
import cv2

def main():
    #* feat add parser
    dataDirPath = getDirAbsPath(sys.argv)
    imgFileExtension = "png"
    files = getFiles(dataDirPath, imgFileExtension)
    dataDirName = dirname(dataDirPath)
    csv_name = f"findmQR-{dataDirName}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    save_path = f"{dataDirPath}/detection/{csv_name}"

    decorder = pb.FactoryFiducial(np.uint8).microqr()
    # decorder = pb.FactoryFiducial(np.uint8).qrcode()
    # detect QRCode in directory images
    for img_file in sorted(files):
        print(f"Processing decord :\t{os.path.basename(img_file)}")
        img_name = os.path.basename(img_file)
        num, res = decodeQR(decorder, img_file)
        convertInfo2CSV(img_name, num, res, save_path)
    

def drawShapeLine(img_path, points):
    color = (255, 0, 0)
    file_name = os.path.basename(img_path)
    dir_name = os.path.dirname(img_path)
    save_dir = f"{dir_name}/detection/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    img = cv2.imread(img_path)
    
    for point in points:
        img = cv2.polylines(img, [np.array(point)], True, color, 5)
    cv2.imwrite(f"{save_dir}/{file_name}", img)

def getFiles(dirPath, extension):
    jpegRobust = ["jpeg", "jpg", "JPEG", "JPG"]
    pngRobust = ["png", "PNG"]
    res = []
    if extension in jpegRobust:
        for e in jpegRobust:
            for file in glob(f"{dirPath}/*.{e}"):
                res.append(file)
    
    elif extension in pngRobust:
        for e in pngRobust:
            for file in glob(f"{dirPath}/*.{e}"):
                res.append(file)
    return res    
    
def getDirAbsPath(cmd_arg):
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
    plist = []
    for code in decorder.detections:
        info = code.message
        points = code.bounds.convert_tuple()
        # convert float num of points to int
        points = [(int(i), int(j)) for i, j in points]
        plist.append(points)
        res[info] = points

    # save new image
    file_name = os.path.basename(img_path)
    dir_name = os.path.dirname(img_path)
    save_dir = f"{dir_name}/"
    drawShapeLine(f"{save_dir}/{file_name}", plist)
    # return dict ex.( {info : [ponts]} )
    return num, res
    
def convertInfo2CSV(img_name, num, dict_, save_path):
    # format : <imgFileName>, <number of detected QRCode>, (x0,y0),(x1,y1),(x2,y2),(x3,y3), <decode string>, ...
    with open(save_path, "a") as f:
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
        # print(f"\tFound:\t{writeLine}")
        writer.writerow(writeLine)

def dirname(path):
    if path[-1] == "/":
        return path.split("/")[-2]
    else:
        return path.split("/")[-1]

if __name__=="__main__":
    main()