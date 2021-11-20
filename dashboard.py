# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:27:29 2021

@author: Cédric
"""

import pandas as pd
import streamlit as st
import numpy as np
from functions import *

data = pd.read_csv('Data/sample_to_train.csv')

st.set_page_config(layout='wide')

# Base Sidebar
sb = st.sidebar

col1, col2, col3 = sb.columns([1,6,1])

with col1:
    st.write("")

with col2:
    st.image('https://play-lh.googleusercontent.com/Q83pGT8fHMAx-Db_oaL0dHCY5-dB8nRLrwGolLeEAJSJjIqyfDr-mh8Q9AnnXHZgO8Y',
             use_column_width=True)

with col3:
    st.write("")
    
pages = sb.radio('', ('Home', 'Analyse Exploratoire', 'Paramètres du modèle', 'Dashboard'))    
    
# Pages

if pages=='Home':
    
    st.write('bite')
    
elif pages=='Analyse Exploratoire':
    
    st.write('Analyse')
    
    st.dataframe(data.head(10))
    
    st.plotly_chart(countplot(data, 'TARGET'))
    
    st.table(correlation(data))
    
    choices = st.multiselect(label="Veuillez choisir les variables à afficher",
                             options=correlation(data).index)
    
    st.plotly_chart(distribution(data, choices))
    
    
    
elif pages=='Paramètres du modèle':
    
    sb.markdown("""___""")    
    sb.slider('% de non remboursement minimum detecté ciblé')
    
elif pages=='Dashboard':
    
    c=st.container()
    
    with c:
        col4, col5, col6 = st.columns([3,3,3])

        with col4:
            st.write("")

        with col5:
            st.text_input('Client selection', help='Fill up with a client id')

        with col6:
            st.write("")
        
    st.write("This is outside the container")

    