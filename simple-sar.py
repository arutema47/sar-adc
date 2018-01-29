# -*- coding: utf-8 -*-

import numpy as np

##########
#written by Ken Yoshioka 1/29/2018
#vectorized sar adc model

#adcin = input signal of the ADC
## min=-0.5, max=0.5
## please normalize!!

#bit = number of bits in SAR ADC
#ncomp = noise of the comparator
#ndac = noise of the c-dac
#nsamp = sampling kT/C noise
#radix = radix of the C-DAC

class SAR:
    def __init__(self, adcin, bit, ncomp, ndac, nsamp, radix):
        self.ncomp = ncomp
        self.ndac = ndac
        self.nsamp = nsamp
        self.bit = bit
        self.adcin = adcin
        self.radix = radix
        self.inloop = adcin.shape[0]
        self.cdac = self.dac()
        self.adcout = np.zeros([self.adcin.shape[0]])
    
    def comp(self, compin):
        comptemp = compin + np.random.rand(compin.shape[0])*self.ncomp
        #comp function in vectors
        out = np.maximum(comptemp*10E6, -1)
        out = np.minimum(out, 1)
        return(out)
    
    def dac(self):
        cdac = np.zeros((self.bit,1))
        for i in range(self.bit):
            cdac[i] = np.power(self.radix,(self.bit-1-i))
        cdac = cdac/(sum(cdac)+1) #normalize
        return(cdac)

    def sarloop(self):
        #add noise to input
        self.adcin += np.random.rand(self.adcin.shape[0])*self.nsamp
        #for cycle
        for cyloop in range(self.bit):
            compout = self.comp(self.adcin)
            self.adcin += compout * (-1) * self.cdac[cyloop] 
            self.adcout += np.power(self.radix, self.bit-1-cyloop)*np.maximum(compout, 0)
            print(cyloop)
        return(self.adcout)

adcin = (np.random.rand(10000)-0.5)/2
print(adcin)
adc = SAR(adcin, 10, 0, 0, 0, 2)
adcout=adc.sarloop()

print(adcout)
