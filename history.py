                X = onehotencoder.fit_transform(X).toarray()
                
                X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=42)

# #################Fit model with some validation (cv) samples ##############
                
                model_xgb, train_pred_xgb, test_pred_xgb = OptimiseXGB(X_train, X_test, y_train, y_test)

# Save model to disk
                result_path = app_settings['result_path']
                model_file_name = Region + state + sku + ".Pickle" 
                model_file_path = path.join(result_path, model_file_name)
                save_model_to_disk(model_xgb, model_file_path)

# #################saving the model as pickel file ##############
    
    
    print('____________________________________________________________________________________________')
    print('Training completed')
debugfile('C:/Users/ashok.swarna/OneDrive - Accenture/ISCP_phase2/Source_Code/Train.py', wdir='C:/Users/ashok.swarna/OneDrive - Accenture/ISCP_phase2/Source_Code')

## ---(Mon May 13 11:16:05 2019)---
df = []
df.to_csv

## ---(Wed May 15 10:39:12 2019)---
import os
import pandas
import pandas as pd
df = pd.read_excel("C:\\Users\\ashok.swarna\\OneDrive - Accenture\\ISCP\\ISCP_DATA.xlsx")
sku_group = df.groupby('Product_SKU', as_index=False)
sku_list = sku_group.groups.keys()

region_group = df.groupby('Region', as_index=False)
region_list = region_group.groups.keys()

state_group = df.groupby('State', as_index=False)
state_list = state_group.groups.keys()
df.columns = df.columns.str.replace(' ', '_')
sku_group = df.groupby('Product_SKU', as_index=False)
sku_list = sku_group.groups.keys()

region_group = df.groupby('Region', as_index=False)
region_list = region_group.groups.keys()

state_group = df.groupby('State', as_index=False)
state_list = state_group.groups.keys()
def load_conditional_data(Region, state, sku, df):
    
    df1 = df[df['Region'].isin ([Region]) & df['State'].isin([state]) &
             df['Product_SKU'].isin([sku]) ].reset_index(drop=True)
    
    df1.sort_values('Order_date')
    
    return df1
from fbprophet import Prophet

## ---(Wed May 15 15:29:54 2019)---
from fbprophet import Prophet

## ---(Mon May 20 13:34:19 2019)---
from app_settings import read_config
from Utility import load_data
from Utility import load_conditional_data
from os import path
from sklearn.externals import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import statistics
from configparser import ConfigParser
import os

def read_config(filename='config.ini', section='settings'):
    
    os.chdir('../config')
    parser = ConfigParser()
    parser.read(filename)
    configurations = {}
    
    if parser.has_section(section):
        items = parser.items(section)
        for item in items:
            configurations[item[0]] = item[1]
    else:
        raise Exception('{0} not found in {1} file'.format(section, filename))
    return configurations
from app_settings import read_config
from Utility import load_data
from Utility import load_conditional_data
from os import path
from sklearn.externals import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import statistics
final_df = pd.read_csv('../Datasets/CO_UT_new.csv', encoding = 'unicode_escape')
import pandas as pd
final_df = pd.read_csv('../Datasets/CO_UT_new.csv', encoding = 'unicode_escape')