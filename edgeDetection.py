import cv2
import numpy as np
import glob
import time
import argparse

# import matplotlib
# from matplotlib.pyplot import imshow
# from matplotlib import pyplot as plt


# read all files
parser = argparse.ArgumentParser(prog='edgeDetection', usage='%(prog)s',description='', epilog="")
parser.add_argument('-f', nargs='?',const = "",type=str, dest='file', help='')
args = parser.parse_args()


def getImages():
  
    if args.file != None:
          for file in glob.glob(args.file + "/*.jpeg"):
              getEdges(file)  
    else:
        for file in glob.glob("*.jpeg"):
              getEdges(file)  
        

def getEdges(path):
    img = cv2.imread(path)# get image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # get grayscale

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0) # remove noise

    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold) # edge detecteion

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),3) # calc lines based on points

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.7, line_image, 1, 0)
    cv2.imwrite("NEW_" + path,lines_edges)


#cv2.imshow("res",lines_edges) # display result
#cv2.waitKey(0)
getImages()