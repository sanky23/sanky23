###
# Variables are directly accessible:
#   print myvar
# Updating a variable:
#   context.updateVariable('myvar', 'new-value')
# Grid Variables are accessible via the context:
#   print context.getGridVariable('mygridvar')
# Updating a grid variable:
#   context.updateGridVariable('mygridvar', [['list','of'],['lists','!']])
# A database cursor can be accessed from the context (Jython only):
#   cursor = context.cursor()
#   cursor.execute('select count(*) from mytable')
#   rowcount = cursor.fetchone()[0]
###


import pandas as pd

import numpy as np
import random
import string
import boto3
import requests
from io import StringIO

import statsmodels.api as sm


load = pd.read_csv('~/Downloads/weekly1forward_v15.csv',index_col=0)
load.index = pd.to_datetime(load.index)


channels = ["Overall"]



for x in channels:
    load[x +'_Adder'] = pd.to_numeric(load[x + '_Adder'], errors = 'coerce')

    Orders = x + '_Orders'
    load['Orders'] = pd.to_numeric(load[Orders], errors = 'coerce')

    data_orders = pd.DataFrame(load, columns= [Orders])
    orders_length = data_orders[data_orders[Orders] >= 0]

    data = load[load[Orders] >= 0]
    data.index = pd.to_datetime(data.index)

    actuals_length = len(orders_length)
    print(actuals_length)
    total_length = len(load)
    print(total_length)


    from statsmodels.tsa.stattools import adfuller
    def test_stationarity(data):

        rolmean = data.rolling(window=52, center=False).mean()
        rolstd = data.rolling(window=52).std()


    #Perform Dickey-Fuller test:

        dftest = adfuller(data, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value


    test_stationarity(data.Orders)


    data['first_difference'] = data.Orders - data.Orders.shift(1)
    test_stationarity(data.first_difference.dropna(inplace=False))


    data['seasonal_difference'] = data.Orders - data.Orders.shift(52)
    test_stationarity(data.seasonal_difference.dropna(inplace=False))

    data['seasonal_first_difference'] = data.first_difference - data.first_difference.shift(52)
    test_stationarity(data.seasonal_first_difference.dropna(inplace=False))

#####4. Build Model:


    endog_load = data.Orders
    endog = endog_load.astype(float)
    exog_load = data[['Holiday_Week', 'Holiday_Pre_Week', 'WTF_Issue', 'Brand_Pre_Consolidation', 'Significant_Event', 'Marketing_Event', 'Acq_Accounts_Affiliate', 'Acq_Accounts_Direct_Mail', 'Acq_Accounts_Facebook', 'Acq_Accounts_Groupon', 'Acq_Accounts_Other', x+'_Avg_wine_cost', x+'_Avg_Bottle_Price_wrong', x+'_Non_Swapped_Order_pct', x+'_SD_pct', x+'_Pct_Off_pct']]
    exog = exog_load.astype(float)

    model = sm.tsa.statespace.SARIMAX(endog = endog, exog = exog, trend='n', order=(2,1,1), seasonal_order=(0,1,1,52))
    results = model.fit()


# code for parsing model summary in dataframe to export table

    summary = results.summary().tables[1]
    summary = pd.DataFrame(summary, columns= ['variables','coefficient','standard_error','z', 'P>|z|' , 'lower_confidence_interval' ,'upper_confidence_interval'])
    summary = summary.drop([0])
    summary['Channel'] = x

#####5. Make Predictions:

    data_forecast = load.iloc[actuals_length:]
    data = pd.concat([data, data_forecast])
    exog_forecast_load = data_forecast[['Holiday_Week', 'Holiday_Pre_Week', 'WTF_Issue', 'Brand_Pre_Consolidation', 'Significant_Event', 'Marketing_Event', 'Acq_Accounts_Affiliate', 'Acq_Accounts_Direct_Mail', 'Acq_Accounts_Facebook', 'Acq_Accounts_Groupon', 'Acq_Accounts_Other', x+'_Avg_wine_cost', x+'_Avg_Bottle_Price_wrong', x+'_Non_Swapped_Order_pct', x+'_SD_pct', x+'_Pct_Off_pct']]
    exog_forecast = exog_forecast_load.astype(float)

    data['forecast'] = results.predict(endog = endog, exog = exog_forecast, start = actuals_length, end= total_length-1, dynamic= True)

    pred_uc = results.get_forecast(steps = total_length-actuals_length, exog = exog_forecast)
    pred_ci = pred_uc.conf_int(alpha = 0.5)

    pred_ci.index.names = ['WE_Sun']
    print(pred_ci)
    print(data['forecast'] )
    final_data = pd.merge(data, pred_ci, how='left', on=['WE_Sun'])
    final_data['Channel'] = x

    final_data.to_csv('~/Downloads/result_forecast3.csv')


#This empties out the dataframe so the loop doesn't add the previous dataframe and bork up the calcs
    data_orders = pd.DataFrame(columns=data_orders.columns)
