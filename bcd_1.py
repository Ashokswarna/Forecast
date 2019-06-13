# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:46:27 2019

@author: ashok.swarna
"""

import os
from os import path
import pandas as pd
import pickle
import numpy as np
from statsmodels.tsa.arima_model import ARIMA, _arma_predict_out_of_sample
from sklearn.metrics import mean_squared_error
from datetime import datetime
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample
from statistics import *
p_range = range(0, 4)
d_range = range(0, 3)
q_range = range(0, 4)

train_period = [
    ['1/1/2018', '12/31/2018']
]

train_period = [[datetime.strptime(y,'%m/%d/%Y') for y in x] for x in train_period]

validation_period = [
    ['11/1/2018', '12/31/2018']
]

validation_period = [[datetime.strptime(y,'%m/%d/%Y') for y in x] for x in validation_period]

mse_period = [
        ['11/1/2018', '12/31/2018']
    
]

mse_period = [[datetime.strptime(y,'%m/%d/%Y') for y in x] for x in mse_period]



material_values =  ['M303.160.117',
               'M303.260.085',
               'M303.260.096',
               'M303.160.120',
               'M303.560.293',
               'M312.860.027' ]
unit_price = [168.39, 185.6, 168.73, 154.66, 100.10, 118.11]


def is_pvalue_significant(pvalues):
    result = True
    pvalues_counter = 0
    for pvalue in pvalues:
        if pvalues_counter != 0:
            if float(pvalue) > 0.05:
                result = False
                break
        pvalues_counter = pvalues_counter + 1

    return result


def revert_to_order(y_log, x_log, d_order=0):
    if d_order == 0:
        result = np.exp(y_log)
        return result
    else:
        pred_diff = pd.Series(y_log, copy=True)
        pred_diff_cumsum = pred_diff.cumsum()
        pred_log = pd.Series(x_log.iloc[0], index=x_log.index)
        pred_log = pred_log.add(pred_diff_cumsum, fill_value=0)
        result = np.exp(pred_log)
        return result
    
def custom_loss(y_real, y_predicted, df_for_loss, price):
  def loss_temp(y_true, y_pred):
    a = Custom_loss2(y_true,y_pred, df_for_loss, price, avg_sale = 100)
    return a
  return loss_temp

import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.losses
import tensorflow as tf
import keras.backend as K
import itertools 
keras.losses.custom_loss = custom_loss


def Custom_loss3(y_true, y_pred, df_for_loss, price, avg_sale = 100, sample_weight=None, multioutput='uniform_average'):
    
    #y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    #check_consistent_length(y_true, y_pred, sample_weight)
    #df_for_loss['y_pred']=y_pred
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock> 0, 0.9*price*df_for_loss.agg_closing_stock, df_for_loss.agg_closing_stock)
    
    #df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock== 0, price*df_for_loss.y_pred, df_for_loss.agg_closing_stock)
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock== 0, 1.1*price*avg_sale, df_for_loss.agg_closing_stock)
    
    df_for_loss.agg_closing_stock= df_for_loss.agg_closing_stock.astype('float64', raise_on_error = False)
    #normalization
    #df_for_loss.agg_closing_stock=(df_for_loss.agg_closing_stock-df_for_loss.agg_closing_stock.min())/(df_for_loss.agg_closing_stock.max()-df_for_loss.agg_closing_stock.min())
    
    #data=tf.convert_to_tensor(df_for_loss.agg_closing_stock)
    data=df_for_loss.agg_closing_stock
    #y_pred=K.tf.multiply(y_pred,data)
    y_pred=y_pred + data

    
    #output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    #if isinstance(multioutput, string_types):
     #   if multioutput == 'raw_values':
      #      return output_errors
       # elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
        #    multioutput = None
    #return np.average(output_errors, weights=multioutput)
    h = y_pred - y_true
    mse = h * h  
    mean_mse = mean(mse)  
    return mean_mse
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    

def Custom_loss2(y_true, y_pred, df_for_loss, price, avg_sale = 100, sample_weight=None, multioutput='uniform_average'):
    
    #y_type, y_true, y_pred, multioutput = _check_reg_targets(y_true, y_pred, multioutput)
    #check_consistent_length(y_true, y_pred, sample_weight)
    #df_for_loss['y_pred']=y_pred
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock> 0, 0.9*price*df_for_loss.agg_closing_stock, df_for_loss.agg_closing_stock)
    
    #df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock== 0, price*df_for_loss.y_pred, df_for_loss.agg_closing_stock)
    df_for_loss['agg_closing_stock'] = np.where(df_for_loss.agg_closing_stock== 0, 1.1*price*avg_sale, df_for_loss.agg_closing_stock)
    
    df_for_loss.agg_closing_stock= df_for_loss.agg_closing_stock.astype('float64', raise_on_error = False)
    #normalization
    #df_for_loss.agg_closing_stock=(df_for_loss.agg_closing_stock-df_for_loss.agg_closing_stock.min())/(df_for_loss.agg_closing_stock.max()-df_for_loss.agg_closing_stock.min())
    
    data=tf.convert_to_tensor(df_for_loss.agg_closing_stock)
    #y_pred=K.tf.multiply(y_pred,data)
    y_pred=K.tf.add(y_pred,data)

    
    #output_errors = np.average((y_true - y_pred) ** 2, axis=0)
    #if isinstance(multioutput, string_types):
     #   if multioutput == 'raw_values':
      #      return output_errors
       # elif multioutput == 'uniform_average':
            # pass None as weights to np.average: uniform mean
        #    multioutput = None
    #return np.average(output_errors, weights=multioutput)
    h = y_pred - y_true
    mse = K.square(h)  
    mean_mse = K.mean(mse, axis=-1)  
    return mean_mse
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    

def mse_predictions(y_real, y_predicted, df_for_loss, price):
   # if len(y_real) > 4:
       # y_predicted = y_predicted[-5:]
     #   y_predicted1 = y_predicted[-5:]
   # else:
    #    y_predicted = y_predicted[-4:]

    print('Real:', y_real)
    print('Predicted:', y_predicted)
    #mse = mean_squared_error(y_real, y_predicted)
    mse = Custom_loss3(y_real, y_predicted, df_for_loss, price)
    return mse

def evaluate_arima_model(x, y, df_for_loss, price, order, start_period, end_period):
    x_log = transform_data(x)
    x_mat = x_log.as_matrix()

    y_mat = y.as_matrix()

    model = ARIMA(x_mat, order=order)
    model_fit = model.fit(disp=0)

    is_significant = is_pvalue_significant(model_fit.pvalues)
    print('model fit pvalues', model_fit.pvalues)

    if is_significant:
        print('Significant')

        # Keep a track of model parameters that are to be saved for predictions
        params = model_fit.params
        residuals = model_fit.resid
        p = model_fit.k_ar
        q = model_fit.k_ma
        k_exog = model_fit.k_exog
        k_trend = model_fit.k_trend
        intercept = params[0]

        # Predict values for the given steps
        y_predict_log = model_fit.predict(start=start_period, end=end_period, exog=None, dynamic=False)

        p_order, d_order, q_order = order

        # Revert predicted log values to normal scale
        y_predict_real = revert_to_order(y_predict_log, x_log, d_order)

        # select
        y_length = len(x_mat)
        y_pred = y_predict_real[-y_length:]


        # Calculate the MSE for the last few predictions
        x_mat = x.as_matrix()
        mse = mse_predictions(x_mat, y_pred, df_for_loss,price)
    else:
        raise Exception('Insignificant model pvalues')

    return params, residuals, p, q, k_exog, k_trend, intercept, mse, y_predict_log



def evaluate_models(data, p_values, d_values, q_values, df_for_loss, price):
    best_score = float('inf')
    best_cfg = None
    best_params = None
    best_residuals = None
    best_p = None
    best_q = None
    best_k_exog = None
    best_k_trend = None
    best_intercept = None

    # split data into train & validation test
    x, y, mse_sales = data

    # No of predictions includes Train + In-Time Validation weeks
    start_step = 1
    # end_step = 137
    # if len(y) > 4:
    #     end_step = 138

    end_step = len(x) + len(y)


    for p_value in p_values:

        for d_value in d_values:

            for q_value in q_values:

                # split data
                try:
                    params, residuals, p, q, k_exog, k_trend, intercept, mse, y_predict_log = evaluate_arima_model(x, mse_sales, df_for_loss, price,
                                                                                                                   order=(
                                                                                                                       p_value,
                                                                                                                       d_value,
                                                                                                                       q_value), 
                                                                                                                   start_period=start_step,
                                                                                                                   end_period=end_step)
                    print('Order p:%d d:%d q:%d' % (p_value, d_value, q_value))
                    # print(params, residuals, p, q, k_exog, k_trend, intercept)
                    print(params, p, q, k_exog, k_trend, intercept)
                    print('MSE:', mse)

                    # Keep track of best least mse model parameters
                    if mse < best_score:
                        best_score = mse
                        best_cfg = (p_value, d_value, q_value)
                        best_params = params
                        best_residuals = residuals
                        best_p = p
                        best_q = q
                        best_k_exog = k_exog
                        best_k_trend = k_trend
                        best_intercept = intercept

                    print()
                except Exception as e:
                    print('Order p:%d d:%d q:%d' % (p_value, d_value, q_value))
                    print('Failed')
                    print(e)
                    print()
                    pass

    return (
    best_score, best_cfg, best_params, best_residuals, best_p, best_q, best_k_exog, best_k_trend, best_intercept,
    y_predict_log)
    
    
def transform_data(data):
    data_log = np.log(data)
    data_log[data_log == -np.inf] = 0
    data_log[data_log == np.inf] = 0
    return data_log

def find_best_model(models):
    print('BEGIN: Selection of best model across periods')

    best_model_score = float('inf')
    best_model = {}
    for model in models:
        if model['mse'] < best_model_score:
            best_model = model
            best_model_score = model['mse']

    # print(best_model)
    # Find best model on R2
    # best_ols_model = find_best_ols_model(models)
    # best_model['best_ols_params'] = best_ols_model

    print()
    print('END: Selection of best model across weeks')

    return best_model

def save_model_to_disk(file_name, model):
    pickle.dump(model, open(file_name, 'wb'))
    print('%s saved' % file_name)
    
    
def get_unitprice(material, material_values, unit_price):
    for i, j in enumerate(material_values):
        if j == material:
            price = unit_price[i] 
    return(price)
    
def train(filename):
    """
        Trains ARIMA model post least MSE per sku & selects the best model and saves it
    :return: None
    """
    begin = 0
    end = 1

    df = pd.read_csv(r'C:\Users\ashok.swarna\bosch_agg.csv')
    df['To_Date'] = pd.to_datetime(df.To_Date, format='%m/%d/%Y')
    
   # ExcelFile
   #df = pd.read_excel(file_path)
   #df = pd.read_excel(file_path)

    # Columns: Sku, Week, Sales
    material = 'M303.160.117'
    df_1 = df[df['Material'].isin([material])]
    
    material_group = df_1.groupby('Material', as_index=False)
    material_list = material_group.groups.keys()
    

    material_best_model = []

    for material in material_list:
        print()
        print(material)

        # Select SKU to train & validate model
        df_sku = df_1[df_1['Material'].isin([material])]
        

        price = get_unitprice(material, material_values, unit_price)
        period_index = 0
        best_period_models = []

        for tp in train_period:
            print()
            #print('Begin:%d End:%d' % (tp[0], tp[1]))
            print()
# Select SKU data from beginning to end of train period
            df_train_period = df_sku[
                    (df_sku['To_Date'] >= tp[begin]) & (df_sku['To_Date'] <= tp[end])]

            
            df_for_loss = df_train_period[['agg_closing_stock','Total_Issue_quantities']]
            
            # Select SKU data from beginning to end of in-time validation period
            df_validation_period = df_sku[
                (df_sku['To_Date'] >= validation_period[period_index][begin]) & (
                        df_sku['To_Date'] <= validation_period[period_index][end])
                ]

            df_mse_period = df_sku[
                (df_sku['To_Date'] >= mse_period[period_index][begin]) & (
                        df_sku['To_Date'] <= mse_period[period_index][end])
                ]
            print('%d train samples for %d period.' % (len(df_train_period), (period_index + 1)))
            print('%d validation samples for %d period.' % (len(df_validation_period), (period_index + 1)))
            print('%d mse samples for %d period.' % (len(df_mse_period), (period_index + 1)))

            # Select sales data for training & validation
            train_sales = df_train_period['Total_Issue_quantities'].reset_index(drop=True)
            validation_sales = df_validation_period['Total_Issue_quantities'].reset_index(drop=True)
            mse_sales = df_mse_period['Total_Issue_quantities'].reset_index(drop=True)

            train_valid_set = (train_sales, validation_sales, mse_sales)

            # Evaluate best model of selected train period

            best_score, best_cfg, best_params, best_residuals, best_p, best_q, best_k_exog, best_k_trend, best_intercept, y_predict_log = evaluate_models(
                train_valid_set, p_range, d_range, q_range, df_for_loss, price)
            
            #forecast
    
            y_pred_log = _arma_predict_out_of_sample(params=best_params, steps=4, errors=best_residuals,
                                                     p=1, q=1, k_trend= best_k_trend, k_exog= best_k_exog, endog = df_sku.To_Date)
            
            
            best_period_model = {'best_cfg': best_cfg, 'mse': best_score, 'Material': sku, 'week': (period_index + 1),
                                 'residuals': best_residuals, 'p': best_p, 'q': best_q, 'k_exog': best_k_exog,
                                 'k_trend': best_k_trend,
                                 'params': best_params, 'intercept': best_intercept}
            best_period_models.append(best_period_model)
            period_index += 1

        # Select best model in entire period
        best_model = find_best_model(best_period_models)

        # Add to best models list
        material_best_model.append(best_model)
        print('____________________________________________________________________________________________')
        print('____________________________________________________________________________________________')

    # Save model to disk
    model_path = app_settings['model_path']

    file_parts = filename.split('.')
    # model_file_name = file_parts[0] + '_HyperParameters.pickle'
    model_file_name = 'model.pickle'

    model_file_path = path.join(model_path, model_file_name)
    save_model_to_disk(model_file_path, sku_best_model)

    print('Training completed')
    
    
def main():
    train('bosch_agg.csv')
    
if __name__ == '__main__':
    main()