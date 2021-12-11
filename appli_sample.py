# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:38:47 2021

@author: Cédric
"""

import numpy as np
import pandas as pd

df = pd.read_csv('Data/application_train.csv')

n_rows = 50000
df_0 = df[df.TARGET == 0]
df_1 = df[df.TARGET == 1]

n_0 = int(df_0.shape[0]*n_rows/df.shape[0])
n_1 = int(df_1.shape[0]*n_rows/df.shape[0])

new_df_0 = df_0.sample(n_0)
new_df_1 = df_1.sample(n_1)

df_sample = pd.concat([new_df_0, new_df_1])

df_sample.to_csv('Data/train_sample.csv', index=False)

# La même méthode est utilisée sur sample_to_train.csv avec n_rows = 15000
# pour des raisons de taille de fichier sur github
