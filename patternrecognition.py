import math
import cv2
import numpy


def calculaLBP(src, quadrantesX, quadrantesY):
    #int i,j,pixelsQuadrante,inicioX,inicioY,fimX,fimY;
    #print('testttttttttt')
    if len(src.shape) is 2:
        rows,cols = src.shape
        channels=1
    else:
        rows,cols,channels = src.shape

    #print(rows)
    #print(cols)
    #asMat=cv2.cv.fromarray(src)
    histograma=[0]*256
    result=[0]*(256*quadrantesX*quadrantesY)
    #Copia a imagem origem e a seta para o padrao binario dela mesma.
    #CvMat dst;
    if channels>1:
        dst=cv2.cv.CreateMat(rows,cols,cv2.CV_8U)
        dst=cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        src=dst.copy()
    else:
        dst=src.copy()
    dst=lbp(dst)
    for currquadY in range(quadrantesY):
        for currquadX in range(quadrantesX):
            #Definindo comeco e fim do quadrante
            inicioX=int((float(cols)/float(quadrantesX))*float(currquadX))
            fimX=int((float(cols)/float(quadrantesX))*float(currquadX+1))
            inicioY=int((float(rows)/float(quadrantesY))*float(currquadY))
            fimY=int((float(rows)/float(quadrantesY))*float(currquadY+1))
            pixelsQuadrante=(fimX-inicioX)*(fimY-inicioY)
            #Calculando Histograma para o Quadrante
            for i in range(inicioY,fimY):
                for j in range(inicioX,fimX):
                    histograma[int(dst[i, j])]+=1
            #print(histograma)
            #print(len(histograma))
            #print(pixelsQuadrante)
            for i in range(len(histograma)):
                result[currquadY*quadrantesX*256+currquadX*256+i]=float(histograma[i])/float(pixelsQuadrante)
                #result[currquadY*quadrantesX*256+currquadX*256+i]=int(str("{:.6e}".format(float(histograma[i])/float(pixelsQuadrante))))
                #result[currquadY*quadrantesX*256+currquadX*256+i]=int(histograma[i]/pixelsQuadrante)
                #print(float(histograma[i])/float(pixelsQuadrante))
                #histograma=inicializaVetor(256)
                #print('Histograma: '+str(histograma))
            
    #print(result)
    return result

def lbp(src):
    #src=cv2.cv.fromarray(srcnp)
    dst=src.copy()
    #rows,cols,channels = src.shape
    if len(src.shape) is 2:
        rows,cols = src.shape
        channels=1
    else:
        rows,cols,channels = src.shape
    #dst=cv2.cv.fromarray(dstnp)
    decimal=0
    for i in range(1,rows-1):
        for j in range(1, cols-1):
            if src[i-1,j-1] >= src[i,j]:
                decimal+=128
            if src[i-1,j] >= src[i,j]:
                decimal+=64
            if src[i-1,j+1] >= src[i,j]:
                decimal+=32
            if src[i,j-1] >= src[i,j]:
                decimal+=16
            if src[i,j+1] >= src[i,j]:
                decimal+=8
            if src[i+1,j-1] >= src[i,j]:
                decimal+=4
            if src[i+1,j] >= src[i,j]:
                decimal+=2
            if src[i+1,j+1] >= src[i,j]:
                decimal+=1
            dst[i,j] = decimal
            if decimal > 255:
                print('valor errado')
            decimal=0
    #cv2.imwrite("LBP.jpg",numpy.asarray(dst))
    #cv2.imshow("test",dst)
    #cv2.waitKey(0)
    #print(dst)
    return dst

def lbp_native(src):
	dst=src.copy()
	radius = 3
	no_points = 8 * radius
	#Uniform LBP is used 17 
	lbp = local_binary_pattern(src, no_points, radius, method='uniform')
	#Calculate the histogram 19 
	x = itemfreq(lbp.ravel())
	# Normalize the histogram
	hist = x[:, 1]/sum(x[:, 1])
	print(np.array(hist, dtype=np.float32))
	return dst

def distanciaEuclidiana(caracteristicaA, caracteristicaB):

    if len(caracteristicaA)!= len(caracteristicaB):
        print("Caracteristicas com vetor de tamanho diferente! Impossivel comparar")
        return -1
    resultAll=[0]*len(caracteristicaA)
    for i in range(len(resultAll)):
        distanciaPontual=caracteristicaA[i]-caracteristicaB[i]
        resultAll[i]=distanciaPontual*distanciaPontual
    soma=sum(resultAll)
    #print("Soma: "+str(soma))
    return math.sqrt(soma)

def diferenca(caracteristicaA, caracteristicaB):
    if len(caracteristicaA)!= len(caracteristicaB):
        print("Caracteristicas com vetor de tamanho diferente! Impossivel comparar")
        return -1
    resultAll=[0]*len(caracteristicaA)
    for i in range(len(resultAll)):
        distanciaPontual=caracteristicaA[i]-caracteristicaB[i]
        resultAll[i]=abs(distanciaPontual)
    return resultAll

