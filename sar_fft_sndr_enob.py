#!/usr/bin/env python
# coding: utf-8

# # SAR ADC

# In[111]:


class SAR:
    def __init__(self, bit, ncomp, ndac, nsamp, radix):
        self.ncomp = ncomp
        self.ndac = ndac
        self.nsamp = nsamp
        self.bit = bit
        self.radix = radix
        self.cdac = self.dac()
    
    def comp(self, compin):
        #note that comparator output suffers from noise of comp and dac
        comptemp = compin + np.random.randn(compin.shape[0])*self.ncomp + np.random.randn(compin.shape[0])*self.ndac
        
        #comp function in vectors
        out = np.maximum(comptemp*10E6, -1)
        out = np.minimum(out, 1)
        return(out)
    
    def dac(self):
        cdac = np.zeros((self.bit,1))
        for i in range(self.bit):
            cdac[i] = np.power(self.radix,(self.bit-1-i))
        cdac = cdac/(sum(cdac)+1) #normalize to full scale = 1
        #mismatches are tbd
        return(cdac)

    def sarloop(self, adcin):
        #add sampling noise to input first
        adcin += np.random.randn(adcin.shape[0]) * self.nsamp
        adcout = np.zeros_like(adcin)
        
        #loop for sar cycles
        for cyloop in range(self.bit):
            compout = self.comp(adcin)
            adcin += compout * (-1) * self.cdac[cyloop] #update cdac output
            adcout += np.power(self.radix, self.bit-1-cyloop)*np.maximum(compout, 0)
            #print(cyloop)
        return(adcout)

def normalize_input(inp):
    center = np.mean(inp)
    out = inp - center
    
    maxbin = np.max(out) * 2
    out = out / maxbin
    
    return out, center, maxbin


# # Generate Signal

# In[126]:


# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# データのパラメータ
fs = 16384
dt = 1/fs
f1 = 1000    # 周波数
t = np.arange(0, N*dt, dt) # 時間軸
freq = np.linspace(0, 1, N)*fs # 周波数軸


# In[168]:


# offset
offset = 1/(2^12)

# 信号を生成
input = np.sin(2*np.pi*f1*t) #+ 0.001 * np.random.randn(N)


# In[169]:


# ADCを宣言
adc = SAR(8, 1e-3, 0, 0, 2)

# 量子化を実施
adcout = adc.sarloop(input)


# In[170]:


plt.plot(adcout[:100])


# In[171]:


# 高速フーリエ変換
F = np.fft.fft(adcout)

# 振幅スペクトルを計算
Amp = np.power(np.abs(F)[0:int(N/2)-1], 2)
Amp[0] = 0 # cut DC
freq = freq[0:int(N/2)-1]
#Amp = np.abs(F)
#freq = freq

# 正規化
Amp /= max(Amp)


# In[172]:


# グラフ表示
plt.figure()
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 17

plt.plot(freq, Amp, label='|F(k)|')
plt.xlabel('Frequency', fontsize=20)
plt.ylabel('Amplitude', fontsize=20)
plt.yscale('log')
plt.grid()
#leg = plt.legend(loc=1, fontsize=25)
#leg.get_frame().set_alpha(1)
plt.show()


# In[173]:


# SNR計算
sig_bin = np.where(Amp==np.abs(Amp).max())[0]
signal_power = Amp[sig_bin]

noise_power = Amp.sum() - signal_power

SNR = signal_power / noise_power
SNR = 10*np.log10(SNR)

ENOB = (SNR-1.76) / 6.02
print("SNR:", SNR)
print("ENOB:", ENOB)


# In[ ]:





# In[ ]:




