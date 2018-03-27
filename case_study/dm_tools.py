import numpy as np
import pandas as pd


def data_prep():
    df = pd.read_csv('organics.csv')
    
    # Justify why we dropp all these values
    df.drop(['CUSTID', 'DOB', 'EDATE', 'NEIGHBORHOOD','LCDATE', 'BILL', 'AGEGRP1', 'AGEGRP2', 'ORGANICS'], axis=1, inplace=True)
    
    #Fill in mean values for nan in numeric values, explored that AGE, AFFL and LTIME has NaN. 
    numeric = ['AGE','AFFL' ,'LTIME']
    
    #Avoid decimals, use mediam
    
    for i in numeric:
        df[i].fillna(df[i].median(), inplace=True)
    
    #Some values are out of range in AFFL, converting them to mean values
    afflmedian = df['AFFL'].median()
    
    for i in range(len(df['AFFL'])):
        if df['AFFL'][i] < 1 or df['AFFL'][i] > 30:
            # replace all 0, 34, and 31 with median()
            df['AFFL'].replace(df['AFFL'][i],afflmedian, inplace = True)
    
    #Fill in GENDER
    df['GENDER'].fillna('U', inplace = True)
    
    #hot-encoding, categorical to numbers
    df2 = pd.get_dummies(df)
        
    return df2

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_

    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)

    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]

    for i in indices:
       print(feature_names[i], ':', importances[i])