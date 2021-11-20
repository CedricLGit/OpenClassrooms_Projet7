# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:14:55 2021

@author: Cédric
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

@st.cache # Cache pour performances streamlit
def countplot(dataframe, x):
    
    dtype = dataframe[x].dtype
    dataframe[x] = dataframe[x].astype(str) # Convertir en string pour une meilleure visualisation
    
    if x == 'TARGET':
        titre = 'Nombre de prêts selon le défault de remboursement'

    fig = px.histogram(dataframe,
                       x=x,
                       color='TARGET',
                       title=titre)
    
    dataframe[x] = dataframe[x].astype(dtype) # Reconvertir au format d'origine
    
    return fig

@st.cache
def correlation(dataframe):
    
    corr = dataframe.corr()
    top10_corr = abs(corr.TARGET.drop('TARGET')).sort_values(ascending=False).head(10)
    
    return top10_corr

@st.cache
def distribution(dataframe, features):
    
    fig = make_subplots(rows=(len(features)//2)+1, cols=2)
    
    for i,feat in enumerate(features):
        
        if i%2 == 0:
            col=1
        else:
            col=2           
        
        data0 = dataframe[dataframe.TARGET==0][feat]
        data1 = dataframe[dataframe.TARGET==1][feat]
        
        fig2 = ff.create_distplot([data0, data1],
                                  ['Prêt sans défault de paiement', 'Prêt avec défault de paiement'],
                                  show_hist=False, show_rug=False)
        
        fig.add_trace(fig2['data'][0],
                      row=(i//2)+1, col=col)
        fig.add_trace(fig2['data'][1],
                      row=(i//2)+1, col=col)
        
    fig.update_layout(showlegend=False,
                      title_text='Distribution des variables selectionnées')
    
    return fig

0%2
