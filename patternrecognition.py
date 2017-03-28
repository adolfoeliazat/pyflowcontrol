import math
import cv2

def calculaLBP(src, quadrantesX, quadrantesY):
    #int i,j,pixelsQuadrante,inicioX,inicioY,fimX,fimY;
    asMat=cv2.cv.fromarray(src)
    histograma=inicializaVetor(256)
    result=inicializaVetor(256*quadrantesX*quadrantesY)
    #Copia a imagem origem e a seta para o padrao binario dela mesma.
    #CvMat dst;
    if asMat.channels>1:
        dst=cv2.cv.CreateMat(asMat.rows,asMat.cols,cv2.CV_8U)
        dst=cv2.cvtColor(src, cv2.cv.CV_BGR2GRAY)
        src=dst.copy()
    else:
        dst=src.clone()
    dst=lbp(dst)
    for currquadY in range(quadrantesY):
        for currquadX in range(quadrantesX):
            #Definindo comeco e fim do quadrante
            inicioX=int((float(asMat.cols)/float(quadrantesX))*float(currquadX))
            fimX=int((float(asMat.cols)/float(quadrantesX))*float(currquadX+1))
            inicioY=int((float(asMat.rows)/float(quadrantesY))*float(currquadY))
            fimY=int((float(asMat.rows)/float(quadrantesY))*float(currquadY+1))
            pixelsQuadrante=(fimX-inicioX)*(fimY-inicioY)
            #Calculando Histograma para o Quadrante
            for i in range(inicioY,fimY):
                for j in range(inicioX,fimX):
                    histograma[int(dst[i, j])]+=1
            #print(histograma)
            #print(len(histograma))
            print(pixelsQuadrante)
            for i in range(len(histograma)):
                result[currquadY*quadrantesX*256+currquadX*256+i]=float(histograma[i])/float(pixelsQuadrante)
                #print(float(histograma[i])/float(pixelsQuadrante))
                #histograma=inicializaVetor(256)
                #print('Histograma: '+str(histograma))
            histograma=inicializaVetor(256)
    return result

def inicializaVetor(tamanho):
    vetor=[0 for i in range(tamanho)]
    return vetor

def lbp(srcnp):
    src=cv2.cv.fromarray(srcnp)
    dstnp=srcnp.copy()
    dst=cv2.cv.fromarray(dstnp)
    decimal=0
    for i in range(1,dst.rows-1):
        for j in range(1, dst.cols-1):
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
            decimal=0
    return dst

def distanciaEuclidiana(caracteristicaA, caracteristicaB):
    if len(caracteristicaA)!= len(caracteristicaB):
        print("Caracteristicas com vetor de tamanho diferente! Impossivel comparar")
        return -1
    resultAll=inicializaVetor(len(caracteristicaA))
    for i in range(len(resultAll)):
        distanciaPontual=caracteristicaA[i]-caracteristicaB[i]
        resultAll[i]=distanciaPontual
    return math.sqrt(sum(resultAll))
