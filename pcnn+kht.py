import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math
from kht import kht
from timeit import default_timer as timer



def dist_Gausiankernel(link_arrange):
    kernel = np.zeros((link_arrange, link_arrange), np.float)

    center_x=round(link_arrange/2)
    center_y=round(link_arrange/2)
    for i in range(link_arrange):
        for j in range(link_arrange):
            if (i==center_x) and (j==center_y):
                kernel[i,j]=1
            else:
                kernel[i,j]=1/math.sqrt(((i)-center_x)**2+((j)-center_y)**2)
    return kernel;

def get_intensity(source):
    source = cv.normalize(source.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    R = source[:,:,0]
    G = source[:,:,1]
    B = source[:,:,2]
    Intensity = np.divide(R + G + B, 3)

    #normalize image
    S = cv.normalize(Intensity.astype('float'), None, 0.0, 64.0, cv.NORM_MINMAX)  

    return S;

def pcnn_modified(source,Alpha_L=0.1, Alpha_T=0.5, V_T=1.0, W=dist_Gausiankernel(9), Beta=0.1, T_extra=63, Num=10):
    dim = source.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float) + T_extra
    Y_AC = np.zeros( dim, np.float)
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        L = Alpha_L * signal.convolve2d(Y, W, mode='same')
        U = source * (1 + Beta * L)

        YC = 1 - Y      
        T = T - Alpha_T
        T = ((Y*T)*V_T) + (YC*T)

        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    #cv.imshow("Result", Y)
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    #cv.imshow("Result Acumulated", Y_AC)

    #Y = (Y*255).astype(np.uint8)
    cv.imwrite('results/PCNN_modified_result.jpg', Y*255)
    cv.imwrite('results/PCNN_modified_acumulated_result.jpg', Y_AC*255)

    return Y, Y_AC;

def pcnn(source, Alpha_F=0.1, Alpha_L=1.0, Alpha_T=0.3, V_F=0.5, V_L=0.2, V_T=20.0, Beta=0.1, Num=10, W=np.array([0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,], np.float).reshape((3, 3)), M=np.array([0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,], np.float).reshape((3, 3))):
    dim = source.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float)
    Y_AC = np.zeros( dim, np.float)
    
    #normalize image
    S = cv.normalize(source.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    #cv.imshow("Result", Y)
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    #cv.imshow("Result Acumulated", Y_AC)

    #Y = (Y*255).astype(np.uint8)
    cv.imwrite('results/PCNN_original_result.jpg', Y*255)
    cv.imwrite('results/PCNN_original_acumulated_result.jpg', Y_AC*255)

    return Y, Y_AC;


def init_kht(source, K_cluster_min_size=10, K_kernel_min_height=0.002, K_cluster_min_deviation=2.0, K_delta=0.5, K_n_sigmas=2):
    source_copy = source.copy()
    source_copy = (source_copy*255).astype(np.uint8)

    kernel = np.ones((5,5),np.float32)
    filter_result = cv.filter2D(source_copy, cv.CV_8U, kernel)
    edges_result = cv.Canny(filter_result, 80, 200)

    cv.imwrite('results/filter_result.jpg', filter_result)
    cv.imwrite('results/canny_result.jpg', edges_result)

    edges_copy = edges_result.copy()

    return kht(edges_copy, cluster_min_size=K_cluster_min_size, cluster_min_deviation=K_cluster_min_deviation, kernel_min_height=K_kernel_min_height, delta=K_delta, n_sigmas=K_n_sigmas), filter_result;

def showLines_kht(lines, source, lines_count=10):
    if(len(source.shape)>2): height, width, _ = source.shape
    else: height, width = source.shape

    source_copy = source.copy()

    for (rho, theta) in lines[:lines_count]:
        theta = math.radians(theta)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)

        h2 = height/2
        w2 = width/2
        if sin_theta != 0:
            one_div_sin_theta = 1 / sin_theta
            x = (round(0) , round(h2 + (rho + w2 * cos_theta) * one_div_sin_theta))
            y = (round(w2+w2) , round(h2 + (rho - w2 * cos_theta) * one_div_sin_theta))
        else:
            x = (round(w2 + rho), round(0))
            y = (round(w2 + rho), round(h2+h2))
        cv.line(source_copy, x, y, (120,0,255), 1, cv.LINE_AA)

    cv.imwrite('results/KHT_result.jpg', source_copy)

    return source_copy;


def main(argv):
    default_file = 'img/3.jpeg'
    filename = argv[0] if len(argv) > 0 else default_file
    
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imwrite('results/image_gray.jpg', gray)

    all_time=0

    start = timer()
    i_image = get_intensity(src)
    end = timer() 
    print("get_intensity:\t", end - start)
    all_time+=(end-start)

    start = timer()
    result, result2 = pcnn_modified(i_image,T_extra=53)
    end = timer()
    print("pcnn_modified:\t", end - start)
    all_time+=(end-start)

    #start = timer()
    #result3, result4 = pcnn(gray)
    #end = timer()
    #print("pcnn function:\t", end - start)
    #all_time+=(end-start)

    start = timer()
    lines, filtrado = init_kht(result)
    end = timer() 
    print("init_kht:\t", end - start)
    all_time+=(end-start)

    start = timer()
    lines_image = showLines_kht(lines,result*255)
    end = timer()
    print("showLines_kht:\t", end - start)
    all_time+=(end-start)

    print("all_time:\t", all_time)
    
        
    cv.waitKey()
    return 0
        
if __name__ == "__main__":
    main(sys.argv[1:])