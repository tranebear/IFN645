import numpy as np
import pandas as pd

def data_prep():
    # read the veteran dataset
    df = pd.read_csv('datasets/veteran.csv')
    
    # change DemCluster from interval/integer to nominal/str
    df['DemCluster'] = df['DemCluster'].astype(str)
    
    # change DemHomeOwner into binary 0/1 variable
    dem_home_owner_map = {'U':0, 'H': 1}
    df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)
    
    # denote errorneous values in DemMidIncome
    mask = df['DemMedIncome'] < 1
    df.loc[mask, 'DemMedIncome'] = np.nan
    
    # impute missing values in DemAge with its mean
    df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)

    # impute med income using mean
    df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)

    # impute gift avg card 36 using mean
    df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)
    
    # drop ID and the unused target variable
    df.drop(['ID', 'TargetD'], axis=1, inplace=True)
    
    df = pd.get_dummies(df)
    
    return df

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

def visualize_decision_tree(dm_model, feature_names, save_name):
    import pydot
    from io import StringIO
    from sklearn.tree import export_graphviz
    
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file


def preprocess_data(df):
    # Q1.4 and Q6.2
    df = df.drop(['Address', 'Landsize', 'BuildingArea', 'YearBuilt', 'Price', 'Bedroom2', 'SellerG'], axis=1)
    
    # Q1.1
    cols_miss_drop =['Postcode', 'CouncilArea', 'Regionname', 'Propertycount']
    mask = pd.isnull(df['Distance'])

    for col in cols_miss_drop:
        mask = mask | pd.isnull(df[col])

    df = df[~mask]
    
    # Q1.2
    df['Bathroom'].fillna(df['Bathroom'].mean(), inplace=True)
    df['Car'].fillna(df['Car'].mean(), inplace=True)
    
    df['Latitude_nan'] = pd.isnull(df['Lattitude'])
    df['Longtitude_nan'] = pd.isnull(df['Longtitude'])
    df['Lattitude'].fillna(0, inplace=True)
    df['Longtitude'].fillna(0, inplace=True)
    
    # Q6.1. Change date into weeks and months
    df['Sales_week'] = pd.to_datetime(df['Date']).dt.week
    df['Sales_month'] = pd.to_datetime(df['Date']).dt.month
    df = df.drop(['Date'], axis=1)  # drop the date, not required anymore
    
    # Q4
    df = pd.get_dummies(df)
    
    return df
