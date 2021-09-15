#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 22:15:01 2021

@author: Rajeev Kumar Sharma
"""

# Importing Numerical and dataframe library 

import numpy as np
import pandas as pd

# Importing dataset file sample.csv in dataset dataframe variable
dataset = pd.read_csv("sample.csv")

# Importing sklearn preprocessing and compose library function such as OneHotEncoder and ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Applying columnTransformer to given dataset for categorical text data given in sample.csv
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
dataset = np.array(columnTransformer.fit_transform(dataset), dtype = np.str)

# Happy Ending
