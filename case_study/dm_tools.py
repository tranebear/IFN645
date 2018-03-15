import numpy as np
import pandas as pd

def data_prep():
    df = pd.read_csv('organics.csv')
    df.drop(['CUSTID', 'DOB', 'EDATE', 'NEIGHBORHOOD','LCDATE'], axis=1, inplace=True)
    
    #Fill in mean values for nan in numeric values, explored that AGE, AFFL and LTIME has NaN. 
    numeric = ['AGE','AFFL' ,'LTIME']
    
    for i in numeric:
        df[i].fillna(df[i].mean(), inplace=True)
    
    #Some values are out of range in AFFL, converting them to mean values
    afflmean = df['AFFL'].mean()
    
    for i in range(len(df['AFFL'])):
        if df['AFFL'][i] < 1 or df['AFFL'][i] > 30:
            # replace all 0, 34, and 31 with mean()
            df['AFFL'].replace(df['AFFL'][i],afflmean, inplace = True)
        
    return df
