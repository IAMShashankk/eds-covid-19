import numpy as np
import pandas as pd
from sklearn import linear_model
from datetime import datetime
from scipy import signal

reg = linear_model.LinearRegression(fit_intercept=True)

def get_doubling_time_via_regression(in_array):
    '''uses linear regression to approximate the slope'''
    y = np.array(in_array)
    X = np.arange(-1,2).reshape(-1,1)

    assert len(in_array)==3

    reg.fit(X,y)
    intercept = reg.intercept_
    slope = reg.coef_

    return intercept/slope

def savgol_filter(df_input, column ='confirmed', window = 5):
    '''savgol filter. to ensure data structre is kept'''
    degree = 1
    df_result = df_input
    filter_in = df_input[column].fillna(0)

    result = signal.savgol_filter(np.array(filter_in),
                                 window,
                                 degree)
    df_result[column+'_filtered'] = result
    return df_result

def rolling_reg(df_input, col='confirmed'):
    '''df_input -> data frame'''
    days_back = 3
    result = df_input[col].rolling(
                                    window = days_back,
                                    min_periods = days_back,
                                    ).apply(get_doubling_time_via_regression, raw=False)
    return result

def calc_filtered_data(df_input, filter_on='confirmed'):
    '''
        Calculate savgol filter and return merged data frame
    '''
    must_contain = set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), 'Error in calc_filtered_data not all columns in data frame'

    pd_filtered_result = df_input[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter).reset_index()
    #print(pd_filtered_result.head())
    df_output = pd.merge(df_input,pd_filtered_result[['index',filter_on+'_filtered']], on=['index'], how='left')

    return df_output

def calc_doubling_rate(df_input, filter_on = 'confirmed'):
    '''
        Calcualte approximated doubling rate and return merged data frame
    '''
    must_contain = set(['state','country',filter_on])
    #print(must_contain)
    #print(df_input.columns)
    assert must_contain.issubset(set(df_input.columns)), 'Error in calc_doubling_rate not all columns in data frame'

    pd_DR_result = df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()
    #print(df_input.head())

    pd_DR_result = pd_DR_result.rename(columns={filter_on:filter_on+'_DR','level_2':'index'})
    #if filter_on == 'confirmed':
        #df_input = df_input.reset_index()
    #print(df_input.head())
    df_output = pd.merge(df_input,pd_DR_result[['index',filter_on+'_DR']], on=['index'],how='left')

    return df_output

if __name__ == '__main__':
    print('Feature Building Started.')
    #test_data = np.array([2,4,6])
    #result = get_doubling_time_via_regression(test_data)
    #print('the test slope is: '+str(result))

    pd_JH_data = pd.read_csv('../data/processed/COVID_relational_confirmed.csv', sep=';', parse_dates=[0])
    # sorting is required - becasue we are assuming sliding window approch; so we are going top to down from first to last sample step by step
    pd_JH_data = pd_JH_data.sort_values('date',ascending=True).reset_index(drop = True).copy()
    #print(pd_JH_data.columns)
    # Index reset is requried as we dropped the index above while sorting. Otherwise index will be messed up and will get out of order results.
    pd_JH_data = pd_JH_data.reset_index()
    # 1. Calcualte the doubling rate for the non filtered data.
    pd_result_large = calc_doubling_rate(pd_JH_data)
    # 2. Calcualte the filtered data
    pd_result_large = calc_filtered_data(pd_result_large)
    #print(pd_result_large.columns)
    # 3. Calcualte the doublinf rate for the filtered data.
    pd_result_large = calc_doubling_rate(pd_result_large,'confirmed_filtered')
    # Mask the data to NaN which has lower than 100 doubling rate. Result in good visualization
    mask=pd_result_large['confirmed']>100
    pd_result_large['confirmed_filtered_DR']=pd_result_large['confirmed_filtered_DR'].where(mask, other=np.NaN)
    pd_result_large.to_csv('../data/processed/COVID_final_set.csv',sep=';',index=False)
    print(pd_result_large[pd_result_large['country']=='Italy'].tail())
    print('Feature Building Ended.')
