import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math
from kht import kht
from timeit import default_timer as timer

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def getKey(item):
    return item[0];

def getKey2(item):
    return item[1];

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

def toHomogeneous(num1, num2=None, normalization=False):
    if num2 is None:
        result=np.asarray(num1,dtype=np.float)
        result=np.append(result,1)
        return result;
    else:
        if(len(num1)+len(num2)==4):
            num1=np.asarray(num1,dtype=np.float)
            num1=np.append(num1,1)
            num2=np.asarray(num2,dtype=np.float)
            num2=np.append(num2,1)
        result = np.cross(num1,num2)
        if normalization:
            denominador = math.sqrt(result[0]**2 + result[1]**2)
            result[0]=result[0]/denominador
            result[1]=result[1]/denominador
            result[2]=result[2]/denominador
        return result;

def toEuclidian(num1,normalization=False):
    #result=np.asarray(num1,dtype=np.float)
    result=np.true_divide(num1[:2], num1[-1])
    return result;


def fromPolar2Euclidian(rho,theta,height,width):
    theta = math.radians(theta)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    h2 = height/2
    w2 = width/2
    if sin_theta != 0:
        one_div_sin_theta = 1 / sin_theta
        x = (0 , h2 + (rho + w2 * cos_theta) * one_div_sin_theta)
        y = (w2+w2 , h2 + (rho - w2 * cos_theta) * one_div_sin_theta)
    else:
        x = (w2 + rho, 0)
        y = (w2 + rho, h2+h2)
    return x,y;

def fromPolar2Euclidian_v2(rho,theta,height,width):
    theta = math.radians(theta)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    # Convert from KHT to Matplotlib's coordinate system conventions.
    # The KHT implementation assumes row-major memory alignment for
    # images. Also, it assumes that the origin of the image coordinate
    # system is at the center of the image, with the x-axis growing to
    # the right and the y-axis growing down.
    if sin_theta != 0:
        x = (-width / 2, width / 2 - 1)
        y = ((rho - x[0] * cos_theta) / sin_theta, (rho - x[1] * cos_theta) / sin_theta)
    else:
        x = (rho, rho)
        y = (-height / 2, height / 2 - 1)
    x = (x[0] + width / 2, x[1] + width / 2)
    y = (y[0] + height / 2, y[1] + height / 2)

    return x,y;

def fromPolar2Euclidian_v3(rho,theta,height,width):
    theta = math.radians(theta)
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    # Convert from KHT to OpenCV window coordinate system conventions.
    # The KHT implementation assumes row-major memory alignment for
    # images. Also, it assumes that the origin of the image coordinate
    # system is at the center of the image, with the x-axis growing to
    # the right and the y-axis growing down.
    # C++ version
    if sin_theta != 0:
        temp = -width * 0.5 
        p1 = (temp, (rho - temp * cos_theta) / sin_theta)
        temp2 = width * 0.5 - 1 
        p2 = (temp2, (rho - temp2 * cos_theta) / sin_theta)
    else:
        p1 = (rho,-height * 0.5)
        p2 = (rho, height * 0.5 - 1)
    p1 = (width * 0.5 + p1[0] , height * 0.5 + p1[1])
    p2 = (width * 0.5 + p2[0] , height * 0.5 + p2[1])
    return p1,p2;

def fromPolar2Euclidian_v4(rho,theta):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    return pt1,pt2;

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

    kernel = np.ones((5,5),np.float)
    #kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

    filter_result = cv.filter2D(source_copy, cv.CV_8U, kernel)
    # kernel = np.ones((5,5),np.float)
    # filter_result = cv.morphologyEx(filter_result, cv.MORPH_OPEN, kernel)
    # filter_result = cv.erode(filter_result,kernel,iterations = 2)

    edges_result = cv.Canny(filter_result, 80, 200)

    cv.imwrite('results/filter_result.jpg', filter_result)
    cv.imwrite('results/canny_result.jpg', edges_result)

    edges_copy = edges_result.copy()

    start = timer() 
    kht_lines = kht(edges_copy, cluster_min_size=K_cluster_min_size, cluster_min_deviation=K_cluster_min_deviation, kernel_min_height=K_kernel_min_height, delta=K_delta, n_sigmas=K_n_sigmas)
    end = timer()
    print("kht only:\t", end - start)

    return kht_lines, filter_result;

def showLines_kht(lines, source, lines_count=0):
    if lines_count==0: lines_count=len(lines)

    if(len(source.shape)>2): height, width, _ = source.shape
    else: height, width = source.shape

    source_copy = source.copy()

    for (rho, theta) in lines[:lines_count]:
        x,y = fromPolar2Euclidian(rho,theta,height,width)
        x = (int(x[0]),int(x[1]))
        y = (int(y[0]),int(y[1]))
        cv.line(source_copy, x, y, (120,0,255), 2, cv.LINE_AA)

    #cv.imwrite('results/KHT_result.jpg', source_copy)

    return source_copy;

def homogeneous_lines(lines, height, width, lines_count=0):
    homo_lines = np.array([],dtype=np.float)
    collection_lines = []
    qtd_linhas = len(lines)
    if lines_count==0: lines_count=qtd_linhas

    lines = sorted(lines, key=getKey)
    lines = np.asarray(lines,dtype=np.float)

    print('Linhas que a função "homogeneous_lines" recebe:\n',lines)

    for (rho, theta) in lines[:lines_count]:
        rho = np.round(rho,4)
        x,y = fromPolar2Euclidian(rho,theta,height,width)
        # belong_ayy = ayy_transpose.dot(line_ayy)
        homo_lines = np.append(homo_lines,toHomogeneous(x,y))

    homo_lines = np.reshape(homo_lines, (lines_count,3))
    # homo_lines_=np.asarray(homo_lines,dtype=np.float)

    #print("Resultado da conversão rho e theta para linhas homogêneas\n", homo_lines)

    for l in homo_lines[:lines_count]:
        subcollection_lines = np.array([])
        for l2 in homo_lines[:lines_count]:
            if np.array_equal(l,l2):
                subcollection_lines = np.append(subcollection_lines,[0,0,0])
                continue;
            subcollection_lines = np.append(subcollection_lines,toHomogeneous(l,l2))
        subcollection_lines = np.reshape(subcollection_lines, (-1,3))
        collection_lines.append(subcollection_lines)
    collection_lines = np.asarray(collection_lines)
    #print("Resultado do produto cruzado de cada linha por cada linha\n",collection_lines)

    set_list = [set() for i in range(qtd_linhas)]

    for i in range(len(collection_lines)):
        set_list[i].add(tuple(lines[i]))
        for j in range(len(collection_lines[i])):
            if collection_lines[i][j][2]==0: continue
            x=j+1
            line1 = toEuclidian(collection_lines[i][j])
            while x<len(collection_lines[i]):
                if collection_lines[i][x][2]==0:
                    x+=1
                    continue;
                line2 = toEuclidian(collection_lines[i][x])
                distance = np.linalg.norm(line1-line2)
                print(i,j,x,distance)
                if distance < 50:
                    set_list[i].add(tuple(lines[j]))
                    set_list[i].add(tuple(lines[x]))
                x+=1

    print(set_list)

    return lines,homo_lines,set_list;

def init_kmeans(lines, interations=10, qtd_clusters=-1, toPlot=False):
    #print("init_kmeans lines\n",lines)
    if qtd_clusters == -1: qtd_clusters = len(lines)
    #X = list(map(list, lines))
    X = np.asarray(lines, dtype=np.float)

    # qtd_180 = X[:,1]>178
    # qtd_0 = X[:,1]<2
    # #print(X)

    # if np.count_nonzero(qtd_180) > np.count_nonzero(qtd_0):
    #     X[qtd_0, 1]=179.5
    # else:
    #     X[qtd_180, 1]=0

    #print(X)


    #testzim = k_means_labels == k
    ##############################################################################
    # Compute clustering with Means

    k_means = KMeans(init='k-means++', n_clusters=qtd_clusters, n_init=interations)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    ##############################################################################
    # Plot result
    if toPlot:
        colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#ffff00', '#aa5357', '#388406', '#f53cbd', '#6b6ac7', '#29c90e', '#735756', '#2828b5', '#f77c27', '#ee68c0', '#5cbb01', '#835291']
        plt.figure()
        #plt.hold(True)
        for k, col in zip(range(qtd_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        plt.title('KMeans')    
        plt.grid(True)
        plt.show()

    return k_means.cluster_centers_

def init_kmeans2(lines, interations=10, qtd_clusters=-1, toPlot=False):
    #print("init_kmeans lines\n",lines)
    #X = list(map(list, lines))

    sorted_lines = sorted(lines, key=getKey2)
    sorted_lines = np.asarray(sorted_lines, dtype=np.float)
    X = np.asarray(lines, dtype=np.float)
    lines_array = X.copy()

    X[:,0] = 1

    qtd_180 = X[:,1]>176
    qtd_0 = X[:,1]<4

    if np.count_nonzero(qtd_180) > np.count_nonzero(qtd_0):
        i=0;
        for bollean_value in qtd_0:
            if bollean_value:
                X[i,1] = X[i,1] + 180
            i+=1
    else:
        i=0;
        for bollean_value in qtd_180:
            if bollean_value:
                X[i,1] = 0 - (180 - X[i,1])
            i+=1

    # print("######################Lines Transformadas")
    # print(X)


    contador_linhasDiferentes = 1
    for i in range(len(sorted_lines)-1):
        if abs(abs(sorted_lines[i][1]) - abs(sorted_lines[i+1][1])) > 5:
            contador_linhasDiferentes+=1

    if qtd_clusters == -1: qtd_clusters = int(contador_linhasDiferentes / 2 + 1)

    #print(X)


    #testzim = k_means_labels == k
    ##############################################################################
    # Compute clustering with Means

    k_means = KMeans(init='k-means++', n_clusters=qtd_clusters, n_init=interations)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    ##############################################################################
    # Plot result
    if toPlot:
        colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#ffff00', '#aa5357', '#388406', '#f53cbd', '#6b6ac7', '#29c90e', '#735756', '#2828b5', '#f77c27', '#ee68c0', '#5cbb01', '#835291']
        plt.figure()
        #plt.hold(True)
        for k, col in zip(range(qtd_clusters), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], 'w',
                    markerfacecolor=col, marker='.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                    markeredgecolor='k', markersize=6)
        plt.title('KMeans')    
        plt.grid(True)
        plt.show()

    clusters_array = []
    maior_k = -1
    qtd_maior_k = 0
    for num_k in range(qtd_clusters):
        my_members = k_means_labels == num_k
        clusters_array.append(lines_array[my_members,:])
        tamanho_cluster = len(clusters_array[num_k])
        if tamanho_cluster > qtd_maior_k:
            maior_k = num_k
            qtd_maior_k = tamanho_cluster

    # print("######################Array1")
    # print(clusters_array)

    for num_k in range(qtd_clusters):
        if num_k != maior_k:
            abs_result = abs(k_means_cluster_centers[maior_k,1])- abs(k_means_cluster_centers[num_k,1])
            print(abs_result)
            if abs(abs_result)<10:
                clusters_array[maior_k] = np.concatenate((clusters_array[maior_k],clusters_array[num_k]), axis=0)

    # print("######################Array2")
    # print(clusters_array)

    # print("######################Lines Array")
    # print(lines_array)

    # print("######################Lines Direcao")
    # print(clusters_array[maior_k])

    print("K_means2 resultados")
    print(k_means_labels)
    print(k_means_cluster_centers)

    return clusters_array[maior_k], k_means.cluster_centers_[maior_k]

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

    print("Quantas linhas voce deseja encontrar?\n")
    input_linhas = int(input())

    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imwrite('results/image_gray.jpg', gray)

    all_time=0

    start = timer()
    i_image = get_intensity(src)
    end = timer() 
    print("get_intensity:\t", end - start)
    all_time+=(end-start)

    start = timer()
    #source,Alpha_L=0.1, Alpha_T=0.5, V_T=1.0, W=dist_Gausiankernel(9), Beta=0.1, T_extra=63, Num=10
    result, result2 = pcnn_modified(i_image,T_extra=63, Alpha_T=2.5, Num=10)
    end = timer()
    print("pcnn_modified:\t", end - start)
    all_time+=(end-start)

    # start = timer()
    # #source, Alpha_F=0.1, Alpha_L=1.0, Alpha_T=0.3, V_F=0.5, V_L=0.2, V_T=20.0, Beta=0.1, Num=10, W=np.array([0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,], np.float).reshape((3, 3)), M=np.array([0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,], np.float).reshape((3, 3))
    # result3, result4 = pcnn(gray)
    # end = timer()
    # print("pcnn function:\t", end - start)
    # all_time+=(end-start)

    start = timer()
    lines, filtrado = init_kht(result)
    end = timer() 
    print("init_kht:\t", end - start)
    all_time+=(end-start)

    lines = lines[:int(input_linhas*2.5)]

    lines2 = sorted(lines, key=getKey)
    lines2 = np.asarray(lines2,dtype=np.float)
    print(lines2)

    # contador_linhasDiferentes = 1
    # for i in range(len(lines)-1):
    #     if abs(abs(lines2[i][0]) - abs(lines2[i+1][0])) > 7:
    #         contador_linhasDiferentes+=1

    # print(contador_linhasDiferentes)

    start = timer()
    #k_lines = init_kmeans(lines[:contador_linhasDiferentes*2],toPlot=False,qtd_clusters=contador_linhasDiferentes)
    k_lines = init_kmeans(lines,toPlot=False,qtd_clusters=input_linhas+1)
    end = timer()
    print("init_kmeans:\t", end - start)
    all_time+=(end-start)

    k_lines_copy = k_lines.copy()

    linhas_direcao, direcao_cluster_center = init_kmeans2(k_lines_copy,toPlot=False)

    start = timer()
    lines_image = showLines_kht(lines,src)
    direcao_image = showLines_kht(linhas_direcao,src)
    lines_filtred = showLines_kht(k_lines,src)
    end = timer()
    print("showLines_kht:\t", end - start)
    all_time+=(end-start)

    print("all_time:\t", all_time)

    cv.imwrite('results/KHT_result1.jpg', lines_image)
    cv.imwrite('results/KHT_result2.jpg', direcao_image)
    cv.imwrite('results/KHT_result3.jpg', lines_filtred)

    if(len(src.shape)>2): height, width, _ = src.shape
    else: height, width = src.shape

    sorted_lines, homo_lines, set_list = homogeneous_lines(k_lines,height,width)

    # comb3 = toHomogeneous(homo_lines[1],homo_lines[2],normalization=False)
    # comb4 = toHomogeneous(homo_lines[1],homo_lines[6],normalization=False)
    # print(comb3,comb4)

    # comb3_eu = toEuclidian(comb3)
    # comb4_eu = toEuclidian(comb4)
    # print(comb3_eu,comb4_eu)
    # print((np.linalg.norm(comb3_eu-comb4_eu)))

    # specifyLines = [sorted_lines[5],sorted_lines[1],sorted_lines[3]]
    # specifyLines = np.asarray(specifyLines, dtype=np.float32)
    # lines_specify = showLines_kht(specifyLines,src)
    # cv.imwrite('results/KHT_result15.jpg', lines_specify)

    cont = 0
    index_maiorLista = 0
    valor_maiorLista = 0
    for lista in set_list:
        specifyLines = []
        tamanho_Lista = len(lista)
        if valor_maiorLista < tamanho_Lista:
            index_maiorLista = cont
            valor_maiorLista = tamanho_Lista
        for line_tuple in lista:
            specifyLines.append(line_tuple)
        np.asarray(specifyLines, dtype=np.float)
        lines_specify = showLines_kht(specifyLines,src)
        cv.imwrite('results/Homo_group_' + str(cont) + '.jpg', lines_specify)
        cont+=1

    print("############################## Grupo de retas paralelas:\n", set_list[index_maiorLista])

    print("############################## Linhas Direcao:\n", linhas_direcao)
    for tupla in set_list[index_maiorLista]:
        tuplaTransformada = np.asarray(tupla, dtype=np.float)
        linhas_direcao = np.append(linhas_direcao, [tuplaTransformada], axis=0)

    print("############################## Linhas Direcao2:\n", linhas_direcao)

    lines_final = showLines_kht(linhas_direcao,src)
    cv.imwrite('results/KHT_result4.jpg', lines_final)

    cv.waitKey()
    return 0
        
if __name__ == "__main__":
    main(sys.argv[1:])
