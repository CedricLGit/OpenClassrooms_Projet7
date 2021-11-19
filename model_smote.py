# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 02:19:55 2021

@author: Cédric
"""

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
import pickle
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import re
import time

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
    X_train = df_train.drop('TARGET', axis=1)
    X_test = df_test
    
    
    # Encoder en 0/1 les categories avec seulement 2 valeurs
    le = LabelEncoder()
    
    for col, col_type in X_train.dtypes.iteritems():
        if (col_type == 'object') and (len(X_train[col].unique()) == 2):
            le.fit(X_train[col])
            le.transform(X_train[col])
            le.transform(X_test[col])
            
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
        df['INCOME_TO_CREDIT_RATIO'] = df['AMT_INCOME_TOTAL']/df['AMT_CREDIT']
        df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
        df['EXT_SOURCE_MEAN'] = df[['EXT_SOURCE_1',
                                    'EXT_SOURCE_2',
                                    'EXT_SOURCE_3']].mean(axis=1)

    # Equilibrer train et test pour avoir les memes colonnes
    X_train, X_test = X_train.align(X_test, join = 'left', axis=1)     
    
    return X_train, X_test, y_train

def aggreg_num(df, agg_var, df_name):
    
    '''
    Regroupe les données numériques avec count/min/max/mean/sum selon 
    la variable agg_var
    '''
    
    # Garder les variables quantitatives uniquement
    num_df = df.select_dtypes('number').copy()
    
    # Récupérer la variable d'aggregation
    num_df[agg_var] = df[agg_var]
    
    # Enlever les identifiants s'ils ne sont pas la variable d'aggregation
    for col in num_df:
        if (col != agg_var and 'SK_ID' in col):
            num_df = num_df.drop(columns = col)

    # Grouper par variable d'aggregation
    agg = num_df.groupby(agg_var).agg(['count', 'min', 'max', 'mean', 'sum']).reset_index()

    # Créer les noms des colonnes
    columns = []

    for var in agg.columns:
        if var[0] != agg_var:
            columns.append('%s_%s_%s' % (df_name, var[0], var[1]))
        else:
            columns.append(agg_var)
    
    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg

def aggreg_cat(df, agg_var, df_name):
    
    '''
    Regroupe les données qualitatives selon la variable agg_var
    '''
    
    # Selection des categories
    categorical = pd.get_dummies(df.select_dtypes('category').copy())

    # Récupérer la variable d'aggregation
    categorical[agg_var] = df[agg_var]

    # Grouper par variable d'aggregation
    agg = categorical.groupby(agg_var).agg(['sum', 'mean']).reset_index()
    
    # Créer les noms des colonnes
    column_names = []
    
    for var in agg.columns:
        if var[0] != agg_var:
            column_names.append('%s_%s_%s' % (df_name, var[0], var[1]))
        else:
            column_names.append(agg_var)

    agg.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(agg, axis = 1, return_index = True)
    agg = agg.iloc[:, idx]
    
    return agg

def aggreg_client(df, group_vars, df_name):
    
    '''
    Regroupe les données par pret au niveau client. Group_vars doit
    être constitué de la variable d'aggregation et d'identification
    exemple : ['SK_ID_PREV', 'SK_ID_CURR']
    '''
    
    # Aggreger quantitatives
    df_agg_num = aggreg_num(df, group_vars[0], df_name)
    
    # Si il y a des variables qualitatives
    if any(df.dtypes == 'category'):
        df_agg_cat = aggreg_cat(df, group_vars[0], df_name)
    
        # Regrouper les deux df
        df_agg = df_agg_cat.merge(df_agg_num, on=group_vars[0], how='outer')
        
        # Regrouper avec id client
        df_agg = df_agg.merge(df[group_vars], on=group_vars[0], how='left')
        
        # Retirer la variable d'aggregation intermediaire
        df_agg = df_agg.drop(columns=[group_vars[0]])
        
        # Regrouper par client
        df_by_client = aggreg_num(df_agg, group_vars[1], 'CLIENT')
        
    # Sans variables qualitative
    else:
        # Regrouper avec id client
        df_agg = df_agg_num.merge(df[group_vars], on=group_vars[0], how='left')
        
        # Retirer la variable d'aggregation intermediaire
        df_agg = df_agg.drop(columns=[group_vars[0]])   
        
        # Regrouper par client
        df_by_client = aggreg_num(df_agg, group_vars[1], 'CLIENT')
    
    return df_by_client

def preprocess_bureau():
    
    bureau_raw = pd.read_csv('Data/bureau.csv')
    bureau_balance_raw = pd.read_csv('Data/bureau_balance.csv')
    
    bureau_raw = reduce_memory_usage(bureau_raw)
    bureau_balance_raw = reduce_memory_usage(bureau_balance_raw)
    
    num_agg_bb = aggreg_num(bureau_balance_raw, 'SK_ID_BUREAU', 'BUREAU_BALANCE')
    cat_agg_bb = aggreg_cat(bureau_balance_raw, 'SK_ID_BUREAU', 'BUREAU_BALANCE')
    
    for df in [num_agg_bb, cat_agg_bb]:
        bureau_raw = bureau_raw.merge(df, on='SK_ID_BUREAU', how='left')
        
    agg_bureau = aggreg_client(bureau_raw, ['SK_ID_BUREAU', 'SK_ID_CURR'], 'BUREAU')
    
    return agg_bureau

def custom_loss(y_true, y_pred, k):
    
    '''
    On veut minimiser le risque de prêter à un client qui a un risque élevé de
    ne pas rembourser le prêt
    
    1 -> client with difficulty (recall à maximiser)
    0 -> others
    
    k argument pour pondérer le poids des erreurs de type FN
    '''

    if y_true.shape == y_pred.shape:
        
        result = []
        
        for i in np.arange(y_true.shape[0]):
            if y_true.iloc[i] == 1:
                diff = k*(y_true.iloc[i] - y_pred[i])
            else:
                diff = (y_true.iloc[i]-y_pred[i])
            result.append(diff)
            
        result = np.sum(np.absolute(result))
        
    return result

def cleaning(df):
    
    # Supprimer les colonnes qui ont plus de 20% de na
    
    perc = 0.5*df.shape[0]
    
    df_clean = df.dropna(axis=1, thresh=perc).copy()
    
    # Supprimer les colonnes redondantes (corrélation élevées) ?
    
    # Imputer les nans
    
    for col in df_clean.columns[df_clean.isna().any()]:
        df_clean[col].fillna(df_clean[col].mean(), inplace = True)
    
    return df_clean

def grid(data_to_fit, y):
    
    X = cleaning(data_to_fit)
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # modèles
    
    models = {'lgbm': LGBMClassifier(n_jobs=-1,
                              random_state=42,
                              max_depth=10,
                              n_estimators=200,
                              learning_rate=0.02,
                              num_leaves=35),
              'RF': RandomForestClassifier(n_jobs=-1,
                                           random_state=42,
                                           n_estimators=200),
              'XGB': xgb.XGBClassifier(n_jobs=-1,
                                       random_state=42,
                                       n_estimators=200,
                                       learning_rate=0.02,
                                       use_label_encoder=False,
                                       max_depth=10),
              'LR': LogisticRegression(n_jobs=-1,
                                       max_iter=1000,
                                       random_state=42)}
    
    # Score
    
    scoring = {'score_k={}'.format(i): make_scorer(custom_loss, k=i,
                                                  greater_is_better=False)
               for i in [2, 5, 10]}
    
    # Plage de paramètres autour des params trouvés par byaesan optimisation
    # sur la metrique auroc disponibles sur le kaggle suivant  
    # https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
    
    params = {'lgbm': {'clf__colsample_bytree': [0.8, 0.85, 0.9],
                       'clf__subsample': [0.8, 0.85, 0.9],
                       'clf__reg_alpha': [0.03, 0.04, 0.05],
                       'clf__reg_lambda': [0.06, 0.07, 0.08]},
              'RF': {'clf__min_samples_split': [2, 10, 50],
                     'clf__min_samples_leaf': [1, 10, 50],
                     'clf__max_samples': [0.8, 0.85, 0.9],
                     'clf__max_depth': [3, 5, 10]},
              'XGB': {'clf__subsample': [0.8, 0.85, 0.9],
                      'clf__colsample_bytree': [0.8, 0.85, 0.9],
                      'clf__reg_alpha': [0.03, 0.04, 0.05],
                      'clf__reg_lambda': [0.06, 0.07, 0.08]},
              'LR': {'clf__C': [0.05, 0.1, 0.2, 0.5, 1]}}

    # Sampling avec combinaise under/oversampling avec SMOTEENN
    
    pipelines = {}
    
    for key, value in models.items():
        pipelines[key] = imbpipeline([['sampling', SMOTE(n_jobs=-1,
                                                         random_state=42)],
                                      ['scaler', StandardScaler()],
                                      ['clf', value]])
    
    # Gridsearch pour le classifier lgbm
    
    grids = {}
    
    for key in models:
        grids[key] = GridSearchCV(pipelines[key],
                                  param_grid=params[key],
                                  scoring=scoring,
                                  refit=False,
                                  cv=3)
        
    # Results
    
    results = {}
    
    for key, gridsearch in grids.items():
        tstart = time.time()
        results[key] = gridsearch.fit(X_train, y_train).cv_results_
        tstop=time.time()
        print(key, tstop-tstart)
    
    # Refit
    
    refit = {}

    for model in models:
    
        sub_result = pd.DataFrame(results[model])
    
        for score in scoring:
    
            sub_result_byscore = sub_result[sub_result['rank_test_{}'.format(score)] == 1]
            index_min = sub_result_byscore.mean_fit_time.idxmin()
            best_params = sub_result_byscore.loc[index_min]['params']
                
            refit[model+'_'+score] = pipelines[model].set_params(**best_params).fit(X_train, y_train)

    mod_fit = {}

    for model in refit:
    
        mod_fit[model] = Pipeline(refit[model].steps[1:])

    conf = {}

    for model in mod_fit:
    
        y_pred = mod_fit[model].predict(X_test)
    
        conf[model] = confusion_matrix(y_test, y_pred)
        
    return results, mod_fit, conf

##############################################################################

def main():
    
    df_train = pd.read_csv('Data/application_train.csv')
    df_test = pd.read_csv('Data/application_test.csv')
    cc_balance = pd.read_csv('Data/credit_card_balance.csv')
    installments = pd.read_csv('Data/installments_payments.csv')
    cash_balance = pd.read_csv('Data/POS_CASH_balance.csv')
    previous = pd.read_csv('Data/previous_application.csv')
    dic_df = {'CC_BALANCE': cc_balance, 'INSTALLMENTS': installments,
              'CASH_BALANCE': cash_balance, 'PREVIOUS': previous}
    
    bureau = preprocess_bureau()
    df_train, df_test, y_train = preprocess_train_test(df_train, df_test)
    
    for k,v in dic_df.items():
        dic_df[k] = aggreg_client(v, ['SK_ID_PREV', 'SK_ID_CURR'], k)

    for df in [df_train, df_test]:
        
        for k, v in dic_df.items():   
            df = df.merge(v, on='SK_ID_CURR', how='left')
            
        df = df.merge(bureau, on='SK_ID_CURR', how='left')
        
    grid(df_train, y_train)
        
    pass

df_appli_train = pd.read_csv('Data/train_sample.csv')
df_appli_test = pd.read_csv('Data/application_test.csv')
cc_balance = pd.read_csv('Data/credit_card_balance.csv')
installments = pd.read_csv('Data/installments_payments.csv')
cash_balance = pd.read_csv('Data/POS_CASH_balance.csv')
previous = pd.read_csv('Data/previous_application.csv')
bureau = preprocess_bureau()
df_appli_train, df_appli_test, y_appli_train = preprocess_train_test(df_appli_train, df_appli_test)
dic_df = {'CC_BALANCE': cc_balance, 'INSTALLMENTS': installments, 'CASH_BALANCE': cash_balance, 'PREVIOUS': previous}

for k,v in dic_df.items():
    dic_df[k] = reduce_memory_usage(v)
    
for k,v in dic_df.items():
    dic_df[k] = aggreg_client(v, ['SK_ID_PREV', 'SK_ID_CURR'], k)
    
df_appli_train = reduce_memory_usage(df_appli_train)
df_appli_test = reduce_memory_usage(df_appli_test)
bureau = reduce_memory_usage(bureau)


# for dataframe in [df_train_prep, df_test_prep]:
dataframe2 = df_appli_train.copy()
for k, v in dic_df.items():   
    dataframe2 = dataframe2.merge(v, on='SK_ID_CURR', how='left')
    print(k)

del bureau, cc_balance, installments, cash_balance, previous, dic_df, df_appli_test, k, v, df_appli_train

dataframe2 = cleaning(dataframe2)
dataframe2 = reduce_memory_usage(dataframe2)

result2, modfit2, confu2 = grid(dataframe2, y_appli_train)

X2 = dataframe2.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y_appli_train)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

legende2=[]
for mod in modfit2:
    
    y_prob2 = modfit2[mod].predict_proba(X_train2)[:,1]

    precision2_, recall2_, thresh2_ = precision_recall_curve(y_train2, y_prob2)
    legende2.append(mod)

    plt.plot(recall2_, precision2_)
    plt.legend(legende2, loc='best', bbox_to_anchor=(1.5, 1))
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision/recall curve SMOTE')

model = pickle.load(open('selected_model_smote.sav', 'rb'))
 
y_prob2 = model.predict_proba(X_test2)[:,1]
thresholds = set(y_prob2)

score = []

for thresh in thresholds:
    
    y_pred = (y_prob2 > thresh).astype(bool)
    score_tresh = custom_loss(y_test2, y_pred, k=2)
    score.append(score_tresh)


np.where(score == min(score))
score[196]

list(thresholds)[588]

y_pred_fix = (y_prob2 > list(thresholds)[588]).astype(bool)

confusion_matrix(y_test2, y_pred_fix)

data_to_train = dataframe2.join(y_appli_train)
data_to_train.to_csv('Data/sample_to_train.csv', index=False)
# precision2_, recall2_, thresh2_ = precision_recall_curve(y_train2, y_prob2)
    
# for i in np.arange(len(thresh2_)):
#     F2smote.append((1+2**2)*precision2_[i]*recall2_[i]/((2**2)*precision2_[i]+recall2_[i]))
    
# thresh2_[F2smote.index(max(F2smote))]
# precision2_[F2smote.index(max(F2smote))]
# recall2_[F2smote.index(max(F2smote))]

# name2 = 'selected_model_smote.sav'
# pickle.dump(modfit2['XGB_score_k=10'], open(name2, 'wb'))

test_api = list(X_test2.iloc[1,:])