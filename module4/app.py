# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns

app = dash.Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

url = 'https://data.cityofnewyork.us/resource/nwxe-4ae8.json'
trees = pd.read_json(url).fillna('NA') 

fig = px.histogram(trees, x='health', color = 'spc_common' , barmode='group')

app.layout = html.Div(children=[
    html.H1(
        children='2015 Street Tree Census - Tree Data'
        ),
    
    dcc.Dropdown(
        id='my_dropdown',
        options = [
            {'label':'curb_loc', 'value':'curb_loc'},
            {'label':'spc_common', 'value':'spc_common'}
        ],
        value = 'curb_loc'
    ),

    dcc.Graph(
        id='my_graph',
        figure=fig
    )
    
])
@app.callback(
    Output('my_graph', 'figure'),
    Input('my_dropdown', 'value'))

def update_graph(value):
    selection = value
    figure = px.histogram(trees, x='health', color = selection  , barmode='group')
    return figure
    
if __name__ == '__main__':
    app.run_server(debug=True)