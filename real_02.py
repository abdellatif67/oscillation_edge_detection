# edge detection from real oscillations
# Author Mohamed abdellatif
########################################################################
import cv2
import numpy as np
from numpy import array
from time import time
import scipy
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
import peakutils.peak
######################################################################
# prepare code

np.seterr(divide='ignore', invalid='ignore')
t0 = time()
cap = cv2.VideoCapture('r_17_nsg30.avi')

ret, frame = cap.read()
rows, cols = frame.shape[:2] 
#skip some frames
for i in range(100): 
    ret, frame = cap.read()
ref_image = frame[:,:,1] # reference is green

dst = np.zeros((rows,cols,3), np.uint8) 
dst[:]=0
n_frames=200
amp= np.zeros((rows,cols), np.uint8) 
ph = np.zeros((rows,cols), np.uint8) 
cf= np.zeros((rows,cols), np.float) 
sr= np.zeros((n_frames,rows,cols)) 

xshft = []
yshft = []
counter=0

# measure osci pattern
##################################################
while(counter<n_frames):
    	ret, frame = cap.read()
    	if ret == True:
	        sr[counter,:,:] = frame[:,:,1] 
 		image = frame[:,:,1] 
		shift, error, diffphase = register_translation(image, ref_image, 100)
		xshft.append(shift[0])
		yshft.append(shift[1])		
		counter+=1 		
        	if cv2.waitKey(30) & 0xFF == ord('q'):
        	    break
	else:
        	break
cap.release()
####################################################
#test computed pattern
#for k in range(1,n_frames):
#	print xshft[k],yshft[k]
########################################################
# second loop scan one reference image
for v in range(1,rows-1):
   for u in range(1,cols-1):
	sg = sr[:,v,u]
	fd = np.gradient(sg)
	amp[v,u]= np.std(fd)
# now get the highest change signal as reference
ind = np.unravel_index(np.argmax(amp, axis=None), amp.shape)
refs=np.gradient(sr[:,ind[0],ind[1]])

####################################
f = np.fft.fft(refs)		    
af = scipy.fft(refs)
fshift = np.fft.fftshift(f) 
y1 = np.abs(fshift)
############################################
for v in range(1,rows-10):
   for u in range(1,cols-10):
	sg = sr[:,v,u]
	fd = np.gradient(sg)
	if amp[v,u]>2.0:
		f1 = np.fft.fft(fd)
		bf = scipy.fft(fd)
        	fshift1 = np.fft.fftshift(f1)
		y2 = np.abs(fshift1)
	
		cc=np.abs(np.corrcoef(y2,y1))
		c = scipy.ifft(af * scipy.conj(bf))
		ps = (np.argmax(abs(c)))	

		cf[v,u]= cc[1][0]
		ph[v,u]= ps 
		
		if cf[v,u]>.70 :
		        #pk = peakutils.peak.indexes(np.array(fd),thres=7.0/max(fd), min_dist=2)
		        #n=len(pk) 
		    	#for k in range(0,n):
				ux=u#+ int(xshft[int(pk[k])])
				vx=v#+ int(yshft[int(pk[k])])	
									
				dst.itemset((vx,ux,0),255)
				dst.itemset((vx,ux,1),255)
				dst.itemset((vx,ux,2),255)

cv2.imwrite("r_17_nsg30_200_70_2.png",dst)
cv2.imwrite("r_17_nsg30_ref.png",ref_image)
t2 = time()
#print 'time is %f' %(t2-t0)
cv2.destroyAllWindows()	
##################################################



