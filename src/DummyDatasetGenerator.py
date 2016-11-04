import numpy as np
import random 


class TestDataClass:
    def __init__ (self,pxx1,pxx2,pxy,pxz,ampx1,ampx2,ampy,ampz):
#The input parameters are (in order): 
#  number of pixels for x in sensor1
#  number of pixels for x in sensor2
#  number of pixels for y (in sensor1)
#  number of pixels for z (in sensor1)
#  Amplitude of x in sensor1
#  Amplitude of x in sensor2
#  Amplitude of y (in sensor1)
#  Amplitude of z (in sensor1)
        self.pxx1 = pxx1
        self.pxx2 = pxx2
        self.pxy  = pxy
        self.pxz  = pxz
        self.ampx1 = ampx1
        self.ampx2 = ampx2
        self.ampy  = ampy
        self.ampz  = ampz       
    def gettrue (self, smpidx = -1 , smpidy = -1, smpidz = -1 ):
#
# get a true pair.
#
# input parameters: given x,y,z (between 0 and 1)
# output (in order): signal in sensor 1, signal in sensor 2, value of x in sensor 1, value of x in sensor 2 (=value of x in sensor 1), value of y, value of z.
#
        if (smpidx<0):
            smpidx1 = random.random()            
        else:
            smpidx1 = smpidx
        smpidx2 = smpidx1
        if (smpidy<0):
            smpidy = random.random()
        if (smpidz<0):
            smpidz = random.random()
            
        sigx1=self.ampx1 * np.sin( (np.matrix(list(range(self.pxx1)))/self.pxx1+smpidx1) * 2*np.pi )
        sigx2=self.ampx2 * np.sin( (np.matrix(list(range(self.pxx2)))/self.pxx2+smpidx2) * 2*np.pi )
        sigy =self.ampy  * np.sin( (np.matrix(list(range(self.pxy )))/self.pxy +smpidy ) * 2*np.pi )
        sigz =self.ampz  * np.sin( (np.matrix(list(range(self.pxz )))/self.pxz +smpidz ) * 2*np.pi )
        
        sig1 = np.append(sigx1,sigy,axis=1)
        sig2 = np.append(sigx2,sigz,axis=1) 
        
        return sig1, sig2, smpidx1, smpidx2, smpidy, smpidz
        
        
        
    def getfake (self, smpidx1 = -1, smpidx2 = -1 , smpidy = -1, smpidz = -1 ):
#
# get a fake pair.
#
# input parameters: given x1,x2,y,z (between 0 and 1)
# output (in order): signal in sensor 1, signal in sensor 2, value of x in sensor 1, value of x in sensor 2, value of y, value of z.
#
        if (smpidx1<0):
            smpidx1 = random.random()            
        if (smpidx2<0):
            smpidx2 = random.random()            
        if (smpidy<0):
            smpidy = random.random()
        if (smpidz<0):
            smpidz = random.random()
            
        sigx1=self.ampx1 * np.sin( (np.matrix(list(range(self.pxx1)))/self.pxx1+smpidx1) * 2*np.pi )
        sigx2=self.ampx2 * np.sin( (np.matrix(list(range(self.pxx2)))/self.pxx2+smpidx2) * 2*np.pi )
        sigy =self.ampy  * np.sin( (np.matrix(list(range(self.pxy )))/self.pxy +smpidy ) * 2*np.pi )
        sigz =self.ampz  * np.sin( (np.matrix(list(range(self.pxz )))/self.pxz +smpidz ) * 2*np.pi )
        
        sig1 = np.append(sigx1,sigy,axis=1)
        sig2 = np.append(sigx2,sigz,axis=1) 
        
        return sig1, sig2, smpidx1, smpidx2, smpidy, smpidz
    



#
# Use example:
#
#import matplotlib.pyplot as plt
#
#
##Create the class:
#tmpcls=TestDataClass(60,45,30,45,1,2,1.5,2.5)
#
##get a random true signal
#s1,s2,x1,x2,y1,y2 = tmpcls.gettrue()
#
#plt.plot( np.matrix(list(range(len(s1.T)))).T, np.append(s1,s2,axis=0).T )
#plt.show()
#
##get a random fake signal
#s1,s2,x1,x2,y1,y2 = tmpcls.getfake()
#plt.plot( np.matrix(list(range(len(s1.T)))).T, np.append(s1,s2,axis=0).T )
#plt.show()
#



