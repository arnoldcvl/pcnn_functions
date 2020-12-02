import sys
import numpy as np
import cv2 as cv
from scipy import signal

W = [0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,]
W = np.array(W, np.float).reshape((3, 3))
M = W

alpha_slider_max = 255

title_window = 'Buttons'
trackbar_name_Beta = 'Beta'
trackbar_name_Num = 'Num'

trackbar_name_VF = 'VF'
trackbar_name_VL = 'VL'
trackbar_name_VT = 'VT'

trackbar_name_AF = 'AF'
trackbar_name_AL = 'AL'
trackbar_name_AT = 'AT'

def main(argv):
    default_file = 'img/3.jpeg'

    filename = argv[0] if len(argv) > 0 else default_file
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    #normalize image
    global S
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    cv.namedWindow(title_window)
    cv.createTrackbar(trackbar_name_Beta, title_window , 10, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_Num, title_window , 10, alpha_slider_max, on_trackbar)

    cv.createTrackbar(trackbar_name_VF, title_window , 50, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_VL, title_window , 20, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_VT, title_window , 20, alpha_slider_max, on_trackbar)

    cv.createTrackbar(trackbar_name_AF, title_window , 10, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_AL, title_window , 100, alpha_slider_max, on_trackbar)
    cv.createTrackbar(trackbar_name_AT, title_window , 30, alpha_slider_max, on_trackbar)    
    
    on_trackbar(0)

    cv.waitKey()

def on_trackbar(val):

    dim = S.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float)
    Y_AC = np.zeros( dim, np.float) 

    Beta = (cv.getTrackbarPos(trackbar_name_Beta, title_window))/100
    Num = cv.getTrackbarPos(trackbar_name_Num, title_window)

    V_F = (cv.getTrackbarPos(trackbar_name_VF, title_window))/100
    V_L = (cv.getTrackbarPos(trackbar_name_VL, title_window))/100
    V_T = cv.getTrackbarPos(trackbar_name_VT, title_window)

    Alpha_F = (cv.getTrackbarPos(trackbar_name_AF, title_window))/100
    Alpha_L = (cv.getTrackbarPos(trackbar_name_AL, title_window))/100
    Alpha_T = (cv.getTrackbarPos(trackbar_name_AT, title_window))/100
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    cv.imshow("Result", Y)
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    cv.imshow("Result Acumulated", Y_AC)

    #Y = (Y*255).astype(np.uint8)
    cv.imwrite('result.jpg', Y*255)
    cv.imwrite('result Acumulated.jpg', Y_AC*255) 
        
if __name__ == "__main__":
    main(sys.argv[1:])