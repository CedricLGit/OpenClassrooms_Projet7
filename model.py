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
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
    
    bureau = pd.read_csv('Data/bureau.csv')
    bureau_balance = pd.read_csv('Data/bureau_balance.csv')
    
    bureau = reduce_memory_usage(bureau)
    bureau_balance = reduce_memory_usage(bureau_balance)
    
    num_agg_bb = aggreg_num(bureau_balance, 'SK_ID_BUREAU', 'BUREAU_BALANCE')
    cat_agg_bb = aggreg_cat(bureau_balance, 'SK_ID_BUREAU', 'BUREAU_BALANCE')
    
    for df in [num_agg_bb, cat_agg_bb]:
        bureau = bureau.merge(df, on='SK_ID_BUREAU', how='left')
        
    agg_bureau = aggreg_client(bureau, ['SK_ID_BUREAU', 'SK_ID_CURR'], 'BUREAU')
    
    return agg_bureau

def custom_loss(y_true, y_pred, k):
    
    '''
    On veut minimiser le risque de prêter à un client qui a un risque élevé de
    ne pas rembourser le prêt
    
    1 -> client with difficulty (recall à maximiser)
    0 -> others (precision à prendre en compte)
    
    k argument pour pondérer le poids des erreurs de type FN
    '''

    if y_true.shape == y_pred.shape:
        
        result = []
        
        for i in np.arange(y_true.shape[0]):
            if y_true[i] == 1:
                diff = k*(y_true[i] - y_pred[i])
            else:
                diff = (y_true[i]-y_pred[i])
            result.append(diff)
            
        result = sum(abs(result))
        
    return result

def grid(data_to_fit, y):
    
    X_train, X_test, y_train, y_test = train_test_split(data_to_fit, y)
    
    clf = LGBMClassifier(n_jobs=-1,
                         random_state=42)
    
    scoring = {'score_k{}'.format(i): make_scorer(custom_loss, k=i,
                                                  greater_is_better=False)
               for i in np.arange(1,5)}
    
    # Plage de paramètres autour des params trouvés par byaesan optimisation
    # sur la metrique auroc disponibles sur le kaggle suivant  
    # https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features
    
    params = {'n_estimators': np.arange(8000, 12001, 500),
              'learning_rate': [0.01, 0.02,0.03, 0.04, 0.05], 
              'num_leaves': np.arange(25, 50), 
              'colsample_bytree': np.arange(0.7, 1, 0.05), 
              'subsample': np.arange(0.7, 1, 0.05), 
              'max_depth': np.arange(5, 11), 
              'reg_alpha': [0.02, 0.03, 0.04, 0.05, 0.1], 
              'reg_lambda': np.arange(0.05, 0.11, 0.01), 
              'min_child_weight': [0.01, 0.1, 25, 30, 35, 40, 50, 100]}
    
    grid = GridSearchCV(clf,
                        param_grid=params,
                        scoring=scoring,
                        refit=False,
                        cv=5)
    
    # Sampling X_train pour équilibrer lors de l'apprentissage
    
    nb_defaut = y_train[y_train == 1].shape[0]
    x_train_defaut = X_train[y_train == 1].copy()
    y_train_defaut = y_train[y_train == 1].copy()
    x_train_ok = X_train[y_train == 0].copy().sample(n=nb_defaut)
    y_train_ok = y_train[y_train == 0].copy()
    
    
    # Fit la gridsearch
    
    grid.fit(X_train, y_train)
    
    result = pd.DataFrame(grid.cv_results_)
        
    return result

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
        
    pass

df_train = pd.read_csv('Data/application_train.csv')
df_test = pd.read_csv('Data/application_test.csv')
cc_balance = pd.read_csv('Data/credit_card_balance.csv')
installments = pd.read_csv('Data/installments_payments.csv')
cash_balance = pd.read_csv('Data/POS_CASH_balance.csv')
previous = pd.read_csv('Data/previous_application.csv')
bureau = preprocess_bureau()

y=df_train['TARGET']

dic_df = {'CC_BALANCE': cc_balance, 'INSTALLMENTS': installments, 'CASH_BALANCE': cash_balance, 'PREVIOUS': previous}
df_train, df_test, y_train = preprocess_train_test(df_train, df_test)

for df in [df_train, df_test, cc_balance, installments, cash_balance, previous, bureau]:
    
    reduce_memory_usage(df)

for k,v in dic_df.items():
    dic_df[k] = aggreg_client(v, ['SK_ID_PREV', 'SK_ID_CURR'], k)

for df in [df_train, df_test]:
        
    for k, v in dic_df.items():   
        df = df.merge(v, on='SK_ID_CURR', how='left')
        print(k)
        

print(previous['SK_ID_CURR'])
print(df_train['SK_ID_CURR'])
