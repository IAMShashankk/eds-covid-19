# %load ../src/visualization/visualize.py
import pandas as pd
import numpy as np
import dash
dash.__version__
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
import os
import plotly.express as px

print(os.getcwd())
df_input_large = pd.read_csv('../data/processed/COVID_final_set.csv', sep=';')
df_analyse = pd.read_csv('../data/processed/COVID_full_flat_table.csv', sep=';')
df_SIR_data = pd.read_csv('../data/processed/COVID_SIR_Model_Data.csv', sep=';')
df_Global_data = pd.read_csv('../data/processed/COVID_GlobalDataView.csv', sep=';')



fig = go.Figure()
df_Global_data_filtered = df_Global_data['Total_Confirmed_Cases'] > 1000
df_Global_data_filtered = df_Global_data[df_Global_data_filtered]

fig_bar = px.scatter(df_Global_data_filtered, x="Total_Active_Cases", y="Total_Deaths",
                 size="Total_Confirmed_Cases", color="Total_Recovered", hover_name="Nation",
                 log_x=True, log_y=True, size_max=60, height = 700)
fig_bar.update_layout( title = 'Global View of COVID cases (>1k confirmed cases) - size represent Total Confirmed cases; hover to see full details.', title_x=0.5,
                xaxis = dict(title='Total Active Cases (Log Scale)'),
                yaxis = dict(title='Total Deaths (Log Scale)')
                )

app = dash.Dash()

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'height': '44px'
}

app.layout = html.Div([

    dcc.Markdown('''
            # Applied Data science on COVID-19 DataSet
            Goal of the project is to learn data science concpet by applying CRISP_DM,
            it covers full walkthrough of: automated data gathering, data transformations,
            filtering and machine learning to approximating the doubling time, and
            {static} deployement of responsive dashboard.

            Time-Series : To check and comapre the growth of COVID confirmed cases in different countries.

            Global View : To check and comapre countries on different parameter like : confiemd cases, deaths, recovered and active cases.

            SIR Model : This view reflects the true growth of infection compare to the predicted growth (using SIR model) in different countries.
            '''),


    dcc.Tabs([
        dcc.Tab(label='Time-Series Visualization', style=tab_style, children=[

            dcc.Markdown('''
            ## Multi-Select country for visualization
            '''),

            dcc.Dropdown(
                id = 'country_drop_down',
                options = [{'label':name, 'value':name} for name in df_input_large['country'].unique()],
                value = ['US', 'Germany', 'India'], # default selected values
                multi = True # for allowing multi value selection
            ),

            dcc.Markdown('''
            ## Select Timeline of confirmed COVID-19 cases or the approximated doubling time
            '''),

            dcc.Dropdown(
                id = 'doubling_time',
                options = [
                    {'label':'Timeline Confirmed', 'value':'confirmed'},
                    {'label':'Timeline Confirmed Filtered', 'value':'confirmed_filtered'},
                    {'label':'Timeline Doubling Rate', 'value':'confirmed_DR'},
                    {'label':'Timeline Doubling Rate Filtered','value':'confirmed_filtered_DR'}
                ],
                value = 'confirmed', # default selected values
                multi = False # Not allowing multi value selection
            ),

            dcc.Graph(figure=fig, id='main_window_slope'),

        ]),

        dcc.Tab(label='Global View', style=tab_style, children=[
            dcc.Graph(figure=fig_bar, id='gloabl_chart')
        ]),

        dcc.Tab(label='SIR Model', style=tab_style,  children=[

            dcc.Markdown('''
            ##  Multi-Select country for visualization.
            '''),

            dcc.Dropdown(
                id = 'country_drop_down_sir',
                options = [{'label':name, 'value':name} for name in df_input_large['country'].unique()],
                value = ['Germany', 'US'], # default selected values
                multi = True # for allowing multi value selection
            ),
            dcc.Graph(figure=fig, id='sir_chart')
        ])

    ])
])

@app.callback(
    Output('sir_chart', 'figure'),
    [Input('country_drop_down_sir', 'value')])
def update_figure(country_list):
    traces = []
    if(len(country_list) > 0):
        for each in country_list:
            nonzero_row = (df_analyse[each] > 0).idxmax(1)
            country_data = df_analyse[each][nonzero_row:]
            ydata = np.array(country_data)
            t = np.arange(len(ydata))
            fitted = np.array(df_SIR_data[each])
            #t, ydata, fitted = Handle_SIR_Modelling(ydata)

            traces.append(dict(
                x = t,
                y = ydata,
                mode = 'markers+lines',
                name = each+str(' - Truth'),
                opacity = 0.9
            ))
            traces.append(
            dict(
                x = t,
                y = fitted,
                mode = 'markers+lines',
                name = each+str(' - Simulation'),
                opacity = 0.9
            ))
    return {
        'data': traces,
        'layout': dict(
            width = 1280,
            height = 720,
            title = 'Fit of SIR model for: '+', '.join(country_list),
            xaxis = {
                'title': 'Days', #'Fit of SIR model for '+str(each)+' cases',
                'tickangle': -45,
                'nticks' : 20,
                'tickfont' : dict(size = 14, color = '#7f7f7f')
            },
            yaxis = {
                'title': 'Population Infected',
                'type': 'log'
            }
        )
    }

@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('doubling_time','value')])
def update_figure(country_list, show_doubling):
    ''' Updates figure based on passed coutry list and filter/doubling value'''

    if '_DR' in show_doubling:
        my_yaxis = {'type':'log',
                    'title':'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
                    }
    else:
        my_yaxis = {'type':'log',
                    'title':'Confirmed infected people (source Johns Hopkins CSSE, log-scale)'
                    }

    traces = []
    for each in country_list:

        df_plot = df_input_large[df_input_large['country']==each]

        # To handle case where we have multiple entry for the same country.
        #improvement can be done by removing this call from callback
        if show_doubling =='confirmed_filtered_DR':
            df_plot = df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.mean).reset_index()
        else:
            df_plot = df_plot[['state','country','confirmed','confirmed_filtered','confirmed_DR','confirmed_filtered_DR','date']].groupby(['country','date']).agg(np.sum).reset_index()

        traces.append(dict(
            x = df_plot.date,
            y = df_plot[show_doubling],
            mode = 'markers+lines',
            name = each,
            opacity = 0.9
        ))
    return {
        'data': traces,
        'layout': dict(
            width = 1280,
            height = 720,
            xaxis = {
                'title': 'Timeline',
                'tickangle': -45,
                'nticks' : 20,
                'tickfont' : dict(size = 14, color = '#7f7f7f')
            },
            yaxis = my_yaxis
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
