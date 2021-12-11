# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:27:29 2021

@author: Cédric
"""

import pandas as pd
import streamlit as st
import numpy as np
from functions import *
import requests
import pickle
import matplotlib.pyplot as plt
import re


st.set_page_config(layout='wide')

raw_data = pd.read_csv('Data_streamlit/train_sample.csv')
raw_data_reduce = reduce_memory_usage(raw_data)
raw_data2 = cleaning(raw_data_reduce.copy())
data = preprocess(raw_data2)
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
sample = pd.read_csv('Data_streamlit/new_v3.csv')
sample2 = sample.drop('TARGET', axis=1)
st.set_option('deprecation.showPyplotGlobalUse', False)
exp_expected_value, shap_values = explanation(sample2.drop('SK_ID_CURR', axis=1))
model = pickle.load(open('model_smote_v3.sav', 'rb'))
model['clf'].get_booster().feature_names = sample2.columns.tolist()
shap_feat = pd.DataFrame(shap_values, columns=sample2.drop('SK_ID_CURR', axis=1).columns) 
abs_shap_feat = abs(shap_feat).mean(axis=0).sort_values(ascending=False)
top_feat = abs_shap_feat.index.values[:10]


# Base Sidebar
sb = st.sidebar

col1, col2, col3 = sb.columns([1, 6, 1])

with col2:
    st.image('https://play-lh.googleusercontent.com/Q83pGT8fHMAx-Db_oaL0dHCY5-dB8nRLrwGolLeEAJSJjIqyfDr-mh8Q9AnnXHZgO8Y',
             use_column_width=True)

pages = sb.radio('', ('Home',
                      'Analyse Exploratoire',
                      'Explication du modèle',
                      'Dashboard'))

# Pages

col4, col5, col6 = st.columns([1, 8, 1])
c = st.container()

if pages == 'Home':

    with col5:
        st.markdown('# <center>Application d\'analyse de risque</center>',
                    True)

    # with c:

    st.write('')
    st.markdown('### <div style="text-indent: 30px;"><u> Contexte</u></div>',
                True)
    st.write('')
    st.write('''Cette application répond à la demande de la société financière
             **"Prêt à dépenser"** de mettre en place un outil répondant aux 
             problématiques suivantes :''')
    st.markdown('* Calculer la probabilité qu\'un client rembourse son crédit.\
                \n * Mettre en place un **dashboard interactif** permettant \
                    d\'expliquer le choix d\'attribution d\'un crédit.')
    st.write('')
    st.markdown('### <div style="text-indent: 30px;"><u> Utilisation</u></div>',
                True)
    st.write('')
    st.write('L\'application dispose de trois onglets :')
    st.markdown('* **Analyse Exploratoire:** permettant de visualiser quelques\
                informations sur le dataset.\
                \n * **Explication du modèle :** reprenant les éléments\
                    importants permettant d\expliquer celui-ci. \
                \n * **Dashboard :** qui permettra d\'expliquer le choix \
                    d\'attribution de crédit.')

elif pages == 'Analyse Exploratoire':

    with col5:
        st.markdown('# <center>Analyse Exploratoire</center>', True)

    st.write('')
    st.write('')
    st.write('')
    st.markdown('### <div style="text-indent: 30px;"><u>Apperçu du jeu de données</u></div>',
                True)
    
    col7, col8, col9 = st.columns([3.7,0.3,6])
    with col7:
        
        st.write('')
        st.write('')
        st.write('')
        st.markdown('<div style="text-align: justify;">Le jeu de données est \
                    consitué de plusieurs fichiers articulés entre eux de la \
                    manière ci-contre.\
                    <BR>\
                    <BR>Le fichier initial comprend les informations \
                        principales concernant le prêt demandé ainsi que des \
                        données sur le client. Par exemple nous retrouvons le \
                        montant du prêt demandé et celui pour chaque échéances,\
                        les revenus du client, son âge, etc ...\
                    <BR>\
                    <BR>Les autres fichiers contiennent des informations sur \
                        les prêts et les demandes de prêt précédents. Ces \
                        informations seront regroupées par client afin \
                        d\'entrer en compte lors du calcul du risque de défaut\
                        de remboursement du crédit.</div>',
                    True)
    with col9:
        st.image('https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png',
                 use_column_width=True)
    st.write('')
    st.write('')
    shape = raw_data2.shape
    st.markdown('Voici un apperçu du fichier principal après avoir retiré les\
                variables ayant plus de 50% de valeurs nulles.\
                Celui-ci a une entrée par client et contient:\
                <BR>\
                <BR>\
                <li> '+ str(shape[0]) + ' lignes/clients</li>\
                <li> '+ str(shape[1]) + ' colonnes/informations sur celui-ci</li>',
                True)
    st.write('')
    
    st.dataframe(raw_data2.head(8))
    st.write('')
    st.write('')
    st.markdown('### <div style="text-indent: 30px;"><u>Déséquilibre des classes</u></div>',
                True)

    col10, col11 = st.columns(2)
    with col10:
        st.plotly_chart(countplot(raw_data2, 'TARGET'), True)
    with col11:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.markdown('<div style="text-align: justify;">De par la nature du \
                    problème étudié (risque de défaut de remboursement), nous\
                    pouvons constater un fort déséquilibre des classes.\
                    <BR>La classe 1 correspondant aux clients qui n\'nont pas\
                        rencontré de défaut de paiement représente environ 90%\
                        de nos clients.\
                    <BR>\
                    <BR>Pour de prendre en compte cet effet et pour prendre en\
                        compte l\'aspect métier (à savoir prioriser la\
                        détection du risque de défaut de paiement), la mise\
                        en place d\'une métrique personnalisée sera nécessaire.</div>',
                    True)
        

    st.markdown('### <div style="text-indent: 30px;"><u>Corrélation linéaire</u></div>',
                True)
    st.write('')
    st.markdown('Afin d\'avoir une première idée des variables/informations\
                importantes lors du choix d\'attribution d\'un crédit, nous\
                pouvons regarder les variables qui sont le plus corrélées avec\
                la classe. En voici ci-dessous la liste.')
    st.write('')
    st.table(correlation(data))
    st.write('')
    st.markdown('Nous pouvons regarder plus en détail leur distribution respective\
                en prenant en compte la classe des individus afin d\'identifier\
                un impact potentiel sur l\'attribution d\'un prêt.')
    
    col12, col13 = st.columns([9,1])
    with col12:    
        choices = st.multiselect(label ='Choisir les variables à afficher',
                                 options=recup_col(raw_data2, correlation(data).index))
    with col13:
        st.write('')
        st.write('')
        button = st.button('Show me')

    if button:

        if len(choices) != 0:
            st.plotly_chart(distribution(raw_data2, choices), True)
        else:
            st.markdown('Veuillez sélectionner des variables à étudier.')

elif pages == 'Explication du modèle':

    with col5:
        st.markdown('# <center>Explication du modèle</center>', True)
    st.write('')
    st.markdown('### <div style="text-indent: 30px;"><u>Fonction coût métier</u></div>',
                True)
    st.write('')
    st.write('')
    st.markdown('<div style="text-align: justify;">La problématique à \
                    laquelle le modèle cherche à répondre est la prédiction \
                    d\'un risque de défault de paiement de la part d\'un client.\
                    Pour celà nous allons mettre en place un fonction customisée\
                    qui comparera la prédiction de notre modèle et la classe \
                    réelle de nos individus en pondérant les erreurs observées.\
                    Nous attribuerons un coefficient plus important aux erreurs\
                    lorsque le modèle se trompera sur un client qui a rencontré\
                    des difficultées de paiement sur le jeu d\'entraînement.\
                    <BR>\
                    <BR>Par ailleurs nous utiliserons cette fonction afin d\'optimiser\
                        le treshold de prédiction. Ceci permettra d\'obtenir de \
                        meilleurs résultats comment le montrent les matrices de\
                        confusions ci contre.<div>',
                    True)
    col7, col8, col9 = st.columns([1,3,1])
    with col8:
        st.image('Data_streamlit/confusion_matrix.png')
    st.write('')
    st.write('')
    st.markdown('### <div style="text-indent: 30px;"><u>Analyse du modèle</u></div>',
                True)
    st.write('')
    col10, col11 = st.columns([0.4, 0.6])
    with col10:
        st.markdown('<div style="text-align: justify;">Après avoir entraîné le \
                    modèle, nous pouvons regarder quelles features contribuent \
                    le plus aux prédictions de celui ci.\
                    <BR>\
                    <BR>Dans un premier temps, nous pouvons récupérer les features\
                    importances que nous renvoie le modèle selectionné.\
                    <BR>XGBoost nous proposent deux types de feature importance:\
                        weight qui représente le nombre de fois qu\'une feature\
                        apparaît dans un arbre de décision ainsi que le gain qui\
                        lui représente la moyenne de la réduction de la fonction\
                        coût quand la feature est utilisée au niveau d’un noeud.</div>',
                    True)
        st.write('')            
        choices = st.selectbox(label ='Choisir le type de feature importance',
                               options=['weight', 'gain'],
                               index=1)
    with col11:
        xgb.plot_importance(model['clf'],
                            max_num_features=10,
                            importance_type=choices,
                            xlabel='Feature importance value')
        st.pyplot(bbox_inches='tight')
        plt.clf()
    st.write('')
    st.write('')
    col12, col13 = st.columns([0.6, 0.4])
    with col13:
        st.write('')  
        st.markdown('<div style="text-align: justify;">Nous pouvons ensuite nous\
                    tourner vers SHAP pour expliquer notre modèle.\
                    <BR>En moyennant les valeurs absolues des valeurs de Shap\
                        pour chaque feature, nous pouvons remonter à l\'importance\
                        globale des variables.\
                    <BR>Nous pouvons aussi représenter les valeurs de Shap pour\
                        chaque individu.</div>',
                    True)
        st.write('')  
        choices2 = st.selectbox(label ='Choisir le type de feature importance',
                               options=['dot', 'bar'],
                               index=1)
    with col12:
        shap.summary_plot(shap_values,
                          sample2.drop('SK_ID_CURR', axis=1),
                          max_display=10,
                          plot_type=choices2)
        st.pyplot(bbox_inches='tight')
        plt.clf()

elif pages == 'Dashboard':

    with col5:
        st.markdown('# <center>Dashboard</center>', True)

    with c:
        col7, col8, col9 = st.columns(3)

        with col8:
            st.write("")
            client_id = st.selectbox('Client selection',
                                      options=sample.SK_ID_CURR.sort_values().values)

            data_input = {"input": sample2[sample2.SK_ID_CURR==int(client_id)].drop('SK_ID_CURR', axis=1).values.tolist()}
            headers = {"Content-Type": "application/json"}
            result = requests.get('https://oc-p7-flaskapi.herokuapp.com/predict',
                                  headers=headers,
                                  json=data_input).text
            st.markdown('### <center>SCORE = '+ result+'</center>',
                        True)        
                
            if float(result)>0.19569081:
                st.markdown('<center>Le score est supérieur au threshold fixé.\
                            <BR>Le crédit est <b>refusé</b>.</center>',
                            True)
            else:
                st.markdown('<center>Le score est inférieur au threshold fixé.\
                            <BR>Le crédit est <b>accepté</b>.</center>',
                            True)
        
    
    st.markdown('### <div style="text-indent: 30px;"><u>Données du client</u></div>',
                True)
    st.write('')    
    st.dataframe(data[data.SK_ID_CURR==client_id].drop('TARGET', axis=1))
    st.write('')
    
    st.markdown('### <div style="text-indent: 30px;"><u>Explication de la\
                décision</u></div>',
                True)
    st.write('')
    st.markdown('Afin d\'expliquer l\'attribution du crédit, nous pouvons \
                regarder l\'impact local des variables sur la valeur Shap\
                du client sélectionné. Nous pouvons voir les variables qui\
                impactent le plus la prédiction du modèle pour celui ci.\
                <BR>En rouge nous avons les variables qui contribuent à ce \
                    que la prédiction soit plus élevée que la valeur de base,\
                    et en bleu celles qui ont l\'effet inverse.',
                True)
    idx = sample2[sample2.SK_ID_CURR==int(client_id)].index
    
    shap.force_plot(exp_expected_value, shap_values[idx,:],
                    sample2.drop('SK_ID_CURR', axis=1).columns.tolist(),
                    contribution_threshold=0.025,
                    matplotlib=True)
    st.pyplot(bbox_inches='tight')
    plt.clf()
    st.write('')
    
    st.markdown('### <div style="text-indent: 30px;"><u>Comparaison</u></div>',
                True)
    ref = data.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    commun = shap_feat.copy()
    commun, ref = commun.align(ref, join = 'inner', axis=1) 
            
    values_client = commun.loc[idx]

    top10_pos = values_client.sort_values(idx[0], ascending=False,
                                          axis=1)
    top10_pos = top10_pos.columns[:10]
    
    selection = recup_col(raw_data2, top10_pos)
    selection += recup_col(raw_data2, correlation(data).index)
    selection = set(selection)
    choice = st.multiselect('Choisir les variables à comparer',
                            options=selection)                              
    if len(choice) != 0:
        st.plotly_chart(distribution(raw_data2,
                                      choice,
                                      comparaison=True,
                                      id_client=int(client_id)),
                        use_container_width=True)