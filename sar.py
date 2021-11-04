# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

"""
written by Ken Yoshioka 1/29/2018
vectorized sar adc model



adcin = input signal of the ADC

bit = number of bits in SAR ADC
ncomp = noise of the comparator
ndac = noise of the c-dac
nsamp = sampling kT/C noise
radix = radix of the C-DAC
"""

class SAR:
    def __init__(self, bit, ncomp, ndac, nsamp, radix, addmismatch=False, mismatch=0.01):
        self.ncomp = ncomp
        self.ndac = ndac
        self.nsamp = nsamp
        self.bit = bit
        self.radix = radix
        
        print("Simulating a {} bit SAR ADC".format(bit))
        
        if addmismatch:
            print("Adding capacitor mismatch")
            self.mismatch = mismatch # default 1% mismatch for unit cap
        else:
            print("Capacitor mismatch not included")
            self.mismatch = 0
        
        # create CDAC
        self.cdac = self.dac()
        
    
    def comp(self, compin):
        #note that comparator output suffers from noise of comp and dac
        comptemp = compin + np.random.randn(compin.shape[0])*self.ncomp + np.random.randn(compin.shape[0])*self.ndac
        
        #comp function in vectors
        out = np.maximum(comptemp*10E6, -1)
        out = np.minimum(out, 1)
        return out
    
    def dac(self):
        cdac = np.zeros((self.bit,1))
        for i in range(self.bit):
            cdac[i] = np.power(self.radix,(self.bit-1-i))
            # add mismatch 
            mis = self.mismatch/np.sqrt(cdac[i])
            cdac[i] += cdac[i]*np.random.randn()*mis
        
        #normalize to full scale = 1
        cdac = cdac/(sum(cdac)+(1+np.random.randn()*mis)) 
        return cdac

    def forward(self, adcin, fft=False):
        #add sampling noise to input first
        adcin += np.random.randn(adcin.shape[0]) * self.nsamp
        adcout = np.zeros_like(adcin)
        
        #loop for sar cycles
        for cyloop in range(self.bit):
            compout = self.comp(adcin)
            adcin += compout * (-1) * self.cdac[cyloop] #update cdac output
            adcout += np.power(self.radix, self.bit-1-cyloop)*np.maximum(compout, 0)
        return adcout
    
    def forward_fft(self, adcin, plot=False):
        # inputs analog waveform and returns SNDR + ENOB
        adcout = self.forward(adcin.copy())
        
        if plot:
            print("plotting conversion results")
            plt.plot(adcout[:100])
            plt.show()
        
        # 高速フーリエ変換
        F = np.fft.fft(adcout)

        # 振幅スペクトルを計算
        N = len(adcin)
        Amp = np.power(np.abs(F)[0:int(N/2)-1], 2)
        Amp[0] = 0 # cut DC
        freq = np.linspace(0, 1, N)*N # 周波数軸
        freq = freq[0:int(N/2)-1]
        #Amp = np.abs(F)
        #freq = freq

        # 正規化
        Amp /= max(Amp)
        
        # グラフ表示
        if plot:
            plt.figure()
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['font.size'] = 17

            plt.plot(freq, Amp, label='|F(k)|')
            plt.xlabel('Frequency', fontsize=20)
            plt.ylabel('Amplitude', fontsize=20)
            plt.yscale('log')
            plt.grid()
            plt.show()
            
        # SNR計算
        sig_bin = np.where(Amp==np.abs(Amp).max())[0]
        signal_power = Amp[sig_bin]

        noise_power = Amp.sum() - signal_power

        SNDR = signal_power / noise_power
        SNDR = 10*np.log10(SNDR)

        ENOB = (SNDR-1.76) / 6.02
        print("SNDR:", SNDR)
        print("ENOB:", ENOB)
        return SNDR, ENOB

def normalize_input(inp):
    center = np.mean(inp)
    out = inp - center
    
    maxbin = np.max(out) * 2
    out = out / maxbin
    
    return out, center, maxbin
    
    
if "__name__" == "main":
    # lets test the adc by random vectors..
    adcin = (np.random.rand(10000)-0.5)/2
    print(adcin)
    adc = SAR(10, 0, 0, 0, 2)
    adcout=adc.forward(adcin)
    
    #print it.
    adc.forward_fft(adcin)
