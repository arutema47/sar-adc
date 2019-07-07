# What is this repo?
- Model SAR ADCs

- Fast Simulations using Numpy

- Can add noise for your design optimization!


# usage
```
from SAR import SAR
import numpy as np

#bit = number of bits in SAR ADC
#ncomp = noise of the comparator
#ndac = noise of the c-dac
#nsamp = sampling kT/C noise
#radix = radix of the C-DAC

# make a ideal sin signal
adcin = np.sin(np.arange(0, 1, 0.001))

bit = 8
ncomp = 0.001
ndac = 0
radix = 2
nsamp = 0

myadc = SAR(adcin, bit, ncomp, ndac, nsamp, radix)

adcout=muadc.sarloop()

#print it.
print(adcout)

# fft to analyze the adc SNDR
TODO
```
