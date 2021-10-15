# -*- coding: utf-8 -*-

'''
Ce script et ses fonctions sont inspirés des Kaggle Kernel
de Will Koehrsen disponibles à l'adresse suivante :
https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
ainsi que de Aguiar disponible ici :
https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
'''
    
    
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

##############################################################################

def reduce_memory_usage(df):
    
    '''
    convertir les dtype afin de réduire l\'utilisation de la mémoire
    '''
    
    for col, col_type in df.dtypes.iteritems():
        if col_type == 'object':
            df[col] = df[col].astype('category')
        elif col_type == 'float64':
            df[col] = df[col].astype('float32')
        elif col_type == 'int64':
            df[col] = df[col].astype('int32')
        
    return df

def preprocess_train_test(df_train, df_test):
    
    '''
    Encoder les variables qualitatives
    Aligner train et test set au besoin
    '''
    
    y_train = df_train['TARGET']
    X_train = df_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    X_test = df_test.drop('SK_ID_CURR', axis=1)
    
    
    # Encoder en 0/1 les categories avec seulement 2 valeurs
    le = LabelEncoder()
    
    for col, col_type in X_train.dtypes.iteritems():
        if (col_type == 'object') and (len(X_train[col].unique()) == 2):
            le.fit(X_train[col])
            le.transform(X_train[col])
            le.transforom(X_test[col])
            
    # Encoder en 'OneHot' les variables de type category
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Il y a des valeurs anormales pour 'DAYS_EMPLOYED' que l'on va supprimer  
    X_train['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    X_test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # FE mannuellement
    
    for df in [X_train, X_test]:
        
        df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
        df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED']/df['DAYS_BIRTH']

    # Equilibrer train et test pour avoir les memes colonnes
    X_train, X_test = X_train.align(X_test, join = 'left', axis=1)     
    
    return X_train, X_test, y_train

def aggreg_num(df, agg_var):
    
    '''
    Regroupe les données numériques avec min/max/mean/sum/count? selon 
    la variable agg_var
    '''
    

    
    return agg
    
    
    pass

##############################################################################

df_train = pd.read_csv('Data/application_train.csv')
df_test = pd.read_csv('Data/application_test.csv')
bureau = pd.read_csv('Data/bureau.csv')
bureau_balance = pd.read_csv('Data/bureau_balance.csv')


