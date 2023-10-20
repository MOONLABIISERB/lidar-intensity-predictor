#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:22:23 2023

@author: moonlab
"""

import numpy as np

lol = np.load('./train_data.npy',allow_pickle = True)
print(len(lol))