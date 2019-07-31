#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 01:15:37 2019

@author: realrangel
"""
from matplotlib import pyplot as plt
from PyEMD import EMD 

#emd = EMD()
#test_data = prec_raw['1997':'1999'].values
#IMFs = emd(test_data)
#imf = IMFs[5:7].sum(axis=0)
#plt.plot(test_data, color='black', linewidth=0.5)
#plt.plot(imf, color='red', linewidth=2, linestyle='--')


emd = EMD()
test_data = precip[precip.notna()].values
IMFs = emd(test_data)
imf = IMFs[-4:].sum(axis=0)
plt.plot(test_data, color='black', linewidth=0.5)
plt.plot(imf, color='red', linewidth=2, linestyle='--')
