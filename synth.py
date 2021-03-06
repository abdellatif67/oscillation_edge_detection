# algorithm to extract edge from synth video using HTR + std dev 
#author mohamed abdellatif

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import skimage
import glob
import os
import sys

from numpy import array
from time import time
from scipy import signal, fftpack
from scipy.signal import correlate
from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
###########################################################
def osci(file):
	cap = cv2.VideoCapture(file)
	n_frames=75 # number of processed frames 3 cycle =3 *25 fpc

	for i in range(25): #skip first cycle
	    ret, frame = cap.read()

	rows, cols = frame.shape[:2] 
	ref_image = frame[:,:,1] # reference is the first frame green
	dst = np.zeros((rows,cols,3), np.uint8) 
	dst[:]=0

	amp= np.zeros((rows,cols), np.uint8) 
	ph = np.zeros((rows,cols), np.uint8) 
	cf= np.zeros((rows,cols), np.float) 
	sr= np.zeros((n_frames,rows,cols)) 
	nzc = np.zeros((rows,cols), np.uint8)

	#fd vertical area edge
	refs=np.array([
	30.0,59.0,55.,46.5,31.5,20.0,7.0,-10.5,-25.5,-40.0,-51.5,-59.5,-46.0,-16.5,-1.0,0.5, 0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,
	30.0,59.0,55.,46.5,31.5,20.0,7.0,-10.5,-25.5,-40.5,-51.5,-59.5,-46.0,-16.5,-1.0,0.5, 0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0,
	30.0,59.0,55.,46.5,31.5,20.0,7.0,-10.5,-25.5,-40.0,-51.5,-59.5,-46.0,-16.5,-1.0,0.5, 0,0.0,0.0,0,0.0,0.0,0.0,0.0,0.0
	])
	#fft
	f = np.fft.fft(refs)		    
	af = scipy.fft(refs)
	fshift = np.fft.fftshift(f) 
	y1 = np.abs(fshift)

	# store frame loop
	arr = []
	for counter in range(n_frames):
	   ret, frame = cap.read() 
	   sr[counter,:,:] = frame[:,:,1] 
	   counter+=1
	cap.release()
	ssg= np.zeros(25)
	#####################
	# second loop to compute signals and derivatives
	for v in range(1,rows-1):
	   for u in range(1,cols-1):
		sg = sr[:,v,u]
		fd = np.gradient(sg)
	
		amp[v,u]= np.std(fd)
	
		if amp[v,u]>1.0: # is there a change in derivative?
		    f1 = np.fft.fft(fd)
		    bf = scipy.fft(fd)
		    fshift1 = np.fft.fftshift(f1)
		    y2 = np.abs(fshift1)
		    cc=np.abs(np.corrcoef(y2,y1))
		    c = scipy.ifft(af * scipy.conj(bf))
		    ps = (np.argmax(abs(c)))%25	
		    cf[v,u]= cc[1][0]
		    ph[v,u]= ps 
			 	
	for v in xrange(1, rows-1):
		for u in xrange(1, cols-1):
			if cf[v,u]>.5 :
				difx=np.abs(int(ph[v,u])-int(ph[v,u+1]))%25
				dify=np.abs(int(ph[v,u])-int(ph[v+1,u]))%25
			
				#dif9 =np.abs( int(amp[v-1,u]  )-int(amp[v,u+1]) )
				#dif10=np.abs( int(amp[v,u-1]  )-int(amp[v+1,u]) )
				#dif11=np.abs( int(amp[v-1,u]  )-int(amp[v,u-1]) )
				#dif12=np.abs( int(amp[v,u+1]  )-int(amp[v+1,u]) )
			
				#if (np.abs(dif9-dif10)<=5 and np.abs(dif11-dif12)<=5 and np.abs(dif9-dif11)<=5 ) or 
				if((difx>=11) and (difx<=13)) or ((dify>=11) and (dify<=13)) :
						
					dst.itemset((v,u,0),255)
			    		dst.itemset((v,u,1),255)
			    		dst.itemset((v,u,2),255)
	cv2.imwrite(file.rsplit( ".", 1 )[ 0 ]+"05.png",dst)
	cv2.destroyAllWindows()  
#########################################################
txtfiles = []
for file in glob.glob("*.avi"):
    txtfiles.append(file)

for file in glob.glob("*.avi"):
	osci(file)

