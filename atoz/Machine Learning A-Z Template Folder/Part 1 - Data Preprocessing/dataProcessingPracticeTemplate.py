# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:13:04 2019

@author: lenovo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.transform(x_test)

