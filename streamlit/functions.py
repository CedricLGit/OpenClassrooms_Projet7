# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:14:55 2021

@author: Cédric
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
import pickle
import shap
import matplotlib.pyplot as plt
import xgboost as xgb


@st.cache
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

@st.cache
def preprocess(df):
    
    '''
    Encoder les variables qualitatives
    '''    
    
    # Encoder en 0/1 les categories avec seulement 2 valeurs
    le = LabelEncoder()
    
    for col, col_type in df.dtypes.iteritems():
        if (col_type == 'object') and (len(df[col].unique()) == 2):
            le.fit(df[col])
            le.transform(df[col])
            
    # Encoder en 'OneHot' les variables de type category
    df = pd.get_dummies(df)    
    
    return df

@st.cache # Cache pour performances streamlit
def cleaning(df):
    
    # Supprimer les colonnes qui ont plus de 50% de na
    
    perc = 0.5*df.shape[0]
    
    df_clean = df.dropna(axis=1, thresh=perc).copy()
    
    # Il y a des valeurs anormales pour 'DAYS_EMPLOYED' que l'on va supprimer  
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Imputer les nans
    
    numdata = df_clean.select_dtypes('number')
    
    for col in numdata.columns[numdata.isna().any()]:
        df_clean[col].fillna(df_clean[col].mean(), inplace = True)
    
    return df_clean

@st.cache
def countplot(dataframe, x):
    
    dtype = dataframe[x].dtype
    dataframe[x] = dataframe[x].astype(str) # Convertir en string pour une meilleure visualisation

    fig = px.histogram(dataframe,
                       x=x,
                       color='TARGET')
    fig.update_layout(yaxis_title='',
                      title={'text': 'Nombre de clients par classe',
                             'x': 0.5,
                             'y': 0.03,
                             'xanchor': 'center',
                             'yanchor': 'bottom'})
    
    dataframe[x] = dataframe[x].astype(dtype) # Reconvertir au format d'origine
    
    return fig

@st.cache
def correlation(dataframe):
    
    corr = dataframe.corr()
    top10 = abs(corr.TARGET.drop('TARGET')).sort_values(ascending=False).head(10).index
    top10_corr = corr.TARGET.drop('TARGET')[top10].sort_values(ascending=False)
    
    return top10_corr

@st.cache
def distribution(dataframe, features, comparaison=False, id_client=''):
    
    nb_row = (len(features)//2)+1
    
    fig = make_subplots(rows=nb_row, cols=2,
                        subplot_titles=['{}'.format(feat) for feat in features],
                        vertical_spacing=0.15)
    leg_num = False
    leg_cat = False
    
    for i,feat in enumerate(features):
        
        if i%2 == 0:
            col=1
        else:
            col=2
            
        if dataframe[feat].dtype == 'category':
            
            fig2=px.histogram(dataframe,
                              x=feat,
                              color='TARGET')
            fig2.update_xaxes(tickangle=45, type='category',
                              showticklabels=True)
            
            if leg_cat:
                fig2['data'][0]['showlegend'] = False
                fig2['data'][1]['showlegend'] = False
                
            else:
                leg_cat=True
                
            fig.add_trace(fig2['data'][0],
                          row=(i//2)+1, col=col)
            fig.add_trace(fig2['data'][1],
                          row=(i//2)+1, col=col)
            
            
        else:
            if feat == 'DAYS_EMPLOYED':
                data0 = dataframe[(dataframe.TARGET==0) & (dataframe[feat]>-10000)][feat]
                data1 = dataframe[(dataframe.TARGET==1) & (dataframe[feat]>-10000)][feat]
            
            else:    
                data0 = dataframe[dataframe.TARGET==0][feat]
                data1 = dataframe[dataframe.TARGET==1][feat]
            fig2 = ff.create_distplot([data0, data1],
                                      ['Prêt sans défault de paiement', 'Prêt avec défault de paiement'],
                                      show_hist=False, show_rug=False)
            if leg_num:
                fig2['data'][0]['showlegend'] = False
                fig2['data'][1]['showlegend'] = False
            else:
                leg_num=True
        
            fig.add_trace(fig2['data'][0],
                          row=(i//2)+1, col=col)
            fig.add_trace(fig2['data'][1],
                          row=(i//2)+1, col=col)
        
        if comparaison == True:
            
            fig.add_vline(x=dataframe[dataframe.SK_ID_CURR == id_client][feat].values[0],
                          line_dash="dash",
                          row=(i//2)+1, col=col)
        
    fig.update_layout(height=400*nb_row)
    
    return fig

@st.cache
def explanation(df_train):
    
    model = pickle.load(open('model_smote_v3.sav', 'rb'))
    explainer = shap.TreeExplainer(model['clf'])
    shap_values = explainer.shap_values(df_train)

    return explainer.expected_value, shap_values

@st.cache
def feature_imp(type_feat):
    
    model = pickle.load(open('model_smote_v3.sav', 'rb'))
    
    if type_feat == 'xgb':
        
        fig = xgb.plot_importance(model['clf'],
                                  max_num_features=10,
                                  importance_type='gain')
    
    return fig

@st.cache
def recup_col(dataframe, liste_col):
    new_col = []
    for col in liste_col:
        if col not in dataframe.columns:
            col = col.replace('_'+col.split('_')[-1], '')
            new_col.append(col)
        else:
            new_col.append(col)
    new_col = list(set(new_col))
    
    return new_col