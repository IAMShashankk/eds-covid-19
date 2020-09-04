import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import subprocess
import os
from datetime import datetime

def store_relational_JH_data():
    data_path = '../data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path)

    pd_data_base = pd_raw.rename(columns={'Country/Region':'country', 'Province/State':'state'})

    pd_data_base['state'] = pd_data_base['state'].fillna('no')

    pd_data_base = pd_data_base.drop(['Lat','Long'],axis=1)

    pd_relational_model = pd_data_base.set_index(['state','country']) \
                                        .T                            \
                                        .stack(level=[0,1])           \
                                        .reset_index()                \
                                        .rename(columns={'level_0':'date',
                                                            0:'confirmed'}
                                                )

    pd_relational_model['date'] = pd_relational_model.date.astype('datetime64[ns]')
    pd_relational_model.to_csv('../data/processed/COVID_relational_confirmed.csv',sep=';', index = False)
    #print(pd_relational_model.head())
    print('Number of rows stored - Relational Model: '+str(pd_relational_model.shape[0]))

def store_confimed_data_for_sir():
    data_path = '../data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    pd_raw = pd.read_csv(data_path)
    pd_raw = pd_raw.drop(['Lat','Long','Province/State'],axis=1)
    pd_raw = pd_raw.rename(columns={'Country/Region':'country'})
    pd_flat_table = pd_raw.set_index('country') \
                    .T \
                    .stack(level=[0]) \
                    .reset_index() \
                    .rename(columns={'level_0':'date',
                                    0:'confirmed'}
                                    )
    pd_flat_table['date'] = pd_flat_table.date.astype('datetime64[ns]')
    pd_flat_table = pd.pivot_table(pd_flat_table, values='confirmed', index='date', columns='country', aggfunc=np.sum, fill_value=0).reset_index()
    pd_flat_table.to_csv('../data/processed/COVID_full_flat_table.csv',sep=';',index = False)
    #print(pd_flat_table.tail())
    print('Number of rows stored - Full Flat Table: '+str(pd_flat_table.shape[0]))

def fetch_globaldata_api():
    url_endpoint = 'https://api.smartable.ai/coronavirus/stats/global'
    headers = {
        # Request headers
        'Cache-Control': 'no-cache',
        'Subscription-Key': '97711648a1aa475c93771173df4ad001',
    }
    response = requests.get(url_endpoint, headers=headers)
    print(response)
    global_dict = json.loads(response.content)
    df = pd.DataFrame() #{'Nation':['A'],'Type':['A'],'Count':[1]}
    for each in global_dict['stats']['breakdowns']:
        total_active_case = ((int(each['totalConfirmedCases'])+int(each['newlyConfirmedCases'])) - (int(each['totalDeaths'])+int(each['newDeaths'])) - (int(each['totalRecoveredCases'])+int(each['newlyRecoveredCases'])))
        dictJSON={'Nation':each['location']['countryOrRegion'],
                 'Total_Confirmed_Cases':int(each['totalConfirmedCases'])+int(each['newlyConfirmedCases']),
                 'Total_Deaths':int(each['totalDeaths'])+int(each['newDeaths']),
                 'Total_Recovered':int(each['totalRecoveredCases'])+int(each['newlyRecoveredCases']),
                 'Total_Active_Cases': int(total_active_case) if total_active_case > 0 else 0 }
        df = df.append(dictJSON, ignore_index=True)
    df.to_csv('../data/processed/COVID_GlobalDataView.csv',sep=';', index = False)

if __name__ == '__main__':
    print('Process Pipeline Started.')
    store_relational_JH_data()
    store_confimed_data_for_sir()
    fetch_globaldata_api()
    print('Process Pipeline Ended.')
