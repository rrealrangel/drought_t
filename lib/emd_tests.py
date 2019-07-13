#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 01:15:37 2019

@author: realrangel
"""
from matplotlib import pyplot as plt
from PyEMD import EMD

#emd = EMD()
#data = prec_raw['1997':'1999'].values
#IMFs = emd(data)
#imf = IMFs[5:7].sum(axis=0)
#plt.plot(data, color='black', linewidth=0.5)
#plt.plot(imf, color='red', linewidth=2, linestyle='--')


emd = EMD()
data = data_subts[data_subts.notnull()].values
IMFs = emd(data)
imf = IMFs[-1:].sum(axis=0)
plt.plot(data, color='black', linewidth=0.5)
plt.plot(imf, color='red', linewidth=2, linestyle='--')
