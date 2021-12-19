from math import trunc
from networkx.algorithms import components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from plotly.subplots import make_subplots
import networkx as nx
import networkx.algorithms.bipartite as bipartite

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.offline as py
from plotly.graph_objs import *
import plotly.graph_objects as go

app = dash.Dash(__name__)

#Constants
date_minusyear = datetime.now() - relativedelta(years=1)
relevant = ['Year', 'Month', 'UserId', 'SMSB_AE','Client', 'Account Abbr', 'Hours', 'Date']

#IMPORTDATA
rawpath = 'https://raw.githubusercontent.com/mtelab1/CUNY_DATA_608/master/Final%20Project/workinglog.csv'
jmraw = pd.read_csv(rawpath, parse_dates=True)
jmraw  = jmraw.drop(['Month','Year'], axis=1)
jmraw  = jmraw.rename(columns={"[Year]": "Year", "[Month]": "Month"})
#jmraw = jmraw[jmraw['JobType'] == 2]
#jmraw = jmraw[relevant]
jmraw["UserId"] = jmraw["UserId"].astype(int).apply(lambda x: chr(ord('`')+x))
jmraw["Client"]= jmraw["Client"].astype(int).apply(lambda x: chr(ord('`')+x))
jmraw["Account Abbr"] = jmraw["Account Abbr"].astype(int).apply(lambda x: chr(ord('`')+x))
jmraw["SMSB_AE"] = jmraw["SMSB_AE"].astype(int).apply(lambda x: chr(ord('`')+x))

jmraw = jmraw.loc[jmraw['Hours'] > 0].reset_index()
tstamps = []
weeks = []
date = []
for n in jmraw['Date']:
    datetime = pd.to_datetime(n)
    tstamps.append(datetime)
    weeks.append(datetime.week)
    date.append(datetime.strftime('%b-%Y'))
jmraw['Timestamp'] = tstamps
jmraw['Week'] = weeks
jmraw['Date'] = pd.to_datetime(date)
jmraw = jmraw.sort_values(by = 'Timestamp', ascending = False).reset_index()

#jmagg = jmraw.groupby(['UserId', 'Date','Year', 'Month', 'Client','Account Abbr']).agg(sum).reset_index()


def pareto_plot(df, x=None, y=None, title=None, show_pct_y=False, pct_format='{0:.0%}'):
    xlabel = x
    ylabel = y
    tmp = df.groupby(x).agg(sum).reset_index().sort_values(y, ascending=False)
    x = tmp[x].values
    y = tmp[y].values
    weights = (y / y.sum())*100
    cumsum = weights.cumsum()
    trunc = min(len(cumsum[cumsum<95]), 25)

    x = x[0:20]
    y = y[0:20]
    cumsum = cumsum[0:20]

    trace1 = go.Bar(
        x = x,
        y = y,
        name='num',
        marker=dict(
            color='rgb(34,163,192)'
                   )
    )   
    trace2 = go.Scatter(
        x = x,
        y = cumsum,
        name='Cumulative Percentage',
        yaxis='y2'

    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(trace1)
    fig.add_trace(trace2,secondary_y=True)
    fig.update_layout(showlegend=False, title = title)
    return fig

def set_dataframes(raw):
    date_min = raw['Date'].min()
    date_max = raw['Date'].max()
    jmselect_clients = raw.groupby(['Client']).agg(sum).reset_index()
    jmselect_users = raw.groupby(['UserId']).agg(sum).reset_index()
    return date_min,date_max,jmselect_clients,jmselect_users

def trim_nodes(g, weight=1):
    nodes = []
    for n in g.nodes():
        if g.degree(n) > weight:
            nodes.append(n)
    G2 = g.subgraph(nodes)
    return G2

def trim_edges(g, weight=1):
    l = []
    g2 = g.copy(as_view=False)
    for f, to, edata in g.edges(data=True):
            if edata['weight'] < weight:
                l.append((f,to))
    g2.remove_edges_from(l)
    nodes = []
    for n in g2.nodes():
        if g2.degree(n) > 0:
            nodes.append(n)
    g2 = g2.subgraph(nodes)
    return g2

def draw_network(data):
    client_list = data.groupby(['Client']).agg(sum).reset_index()
    user_list = data.groupby(['UserId']).agg(sum).reset_index()
    graph = nx.Graph()

    for n in range(len(client_list)):
        graph.add_node(client_list['Client'][n],
                Type='Client', Hours=client_list['Hours'][n])

    for n in range(len(user_list)):
        graph.add_node(user_list['UserId'][n], Type='User', Hours=300)

    for n in range(len(data)):
        v = data['UserId'][n]
        m = data['Client'][n]
        w = data['Hours'][n]
        graph.add_weighted_edges_from([(v, m, w)], weight='time')

    ncol = nx.get_node_attributes(graph, 'Type')
    ncol = ['green' if ncol[v] == 'Client' else 'blue' for v in ncol]
    nsize = nx.get_node_attributes(graph, 'Hours')
    nsize = np.array(list(nsize.values()))
    nsize = np.sqrt(nsize)
    nsize = ((nsize/np.max(nsize))*40)
    esize = np.array(list(nx.get_edge_attributes(graph, 'time').values()))
    esize = list(np.sqrt(esize))
    
    spring_3D = nx.spring_layout(graph, dim=3, seed=18)

    Num_nodes = len(graph.nodes)
    nnames = list(graph.nodes)

    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = [spring_3D[i][0] for i in graph.nodes]  # x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in graph.nodes]  # y-coordinates
    z_nodes = [spring_3D[i][2] for i in graph.nodes]  # z-coordinates

    # We also need a list of edges to include in the plot
    edge_list = graph.edges()


    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    # need to fill these with all of the coordiates
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0], spring_3D[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1], spring_3D[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2], spring_3D[edge[1]][2], None]
        z_edges += z_coords

        # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            hoverinfo='none')

    # create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=nsize,
                                        color=ncol,  # color the nodes according to their community
                                        # either green or mageneta
                                        colorscale=['lightgreen', 'magenta'],
                                        line=dict(color='black', width=0.5)),
                            #                        text=club_labels,
                            hovertext=nnames,
                            hoverinfo='text')


    # we need to set the axis for the plot
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    # also need to create the layout for our plot
    layout = go.Layout(title='SMSB Consulting Group Org Chart',
                    #width=1000,
                    #height=1000,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                                yaxis=dict(axis),
                                zaxis=dict(axis),
                                ),
                    margin=dict(t=100),
                    hovermode='closest')

    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    fig_network = go.Figure(data=data, layout=layout)
    return fig_network
    #py.plot(fig, filename='SMSB Consulting Group Org Chart')


def draw_proj_network(data,trim_weight =1 ):
    client_list = data.groupby(['Client']).agg(sum).reset_index()
    user_list = data.groupby(['UserId']).agg(sum).reset_index()
    graph = nx.Graph()

    for n in range(len(client_list)):
        graph.add_node(client_list['Client'][n],
                Type='Client', Hours=client_list['Hours'][n])

    for n in range(len(user_list)):
        graph.add_node(user_list['UserId'][n], Type='User', Hours=300)

    for n in range(len(data)):
        v = data['UserId'][n]
        m = data['Client'][n]
        w = data['Hours'][n]
        graph.add_weighted_edges_from([(v, m, w)], weight='time')

    graph = bipartite.weighted_projected_graph(graph, user_list['UserId'])
    
    graph_lineweight = [v for v in nx.get_edge_attributes(graph, 'weight').values()]
    graph_nodeweight = [v for v in nx.get_node_attributes(graph, 'weight').values()]
    #graph = trim_nodes(graph, weight=trim_weight)
    graph = trim_edges(graph, weight=trim_weight)
    graph_nodeweight = [v for v in dict(graph.degree()).values()]

    
    
    nsize = graph_nodeweight
    esize = graph_lineweight
    ncol = nx.get_node_attributes(graph, 'Type')
    ncol = ['green' if ncol[v] == 'Client' else 'blue' for v in ncol]
    
    spring_3D = nx.spring_layout(graph, k=None, pos=None, fixed=None, iterations=50, threshold=0.001, weight='weight', scale=.05, center=None, dim=3, seed=44)
    
     
    Num_nodes = len(graph.nodes)
    nnames = list(graph.nodes)

    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = [spring_3D[i][0] for i in graph.nodes]  # x-coordinates of nodes
    y_nodes = [spring_3D[i][1] for i in graph.nodes]  # y-coordinates
    z_nodes = [spring_3D[i][2] for i in graph.nodes]  # z-coordinates

    # We also need a list of edges to include in the plot
    edge_list = graph.edges()


    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    # need to fill these with all of the coordiates
    for edge in edge_list:
        #format: [beginning,ending,None]
        x_coords = [spring_3D[edge[0]][0], spring_3D[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [spring_3D[edge[0]][1], spring_3D[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [spring_3D[edge[0]][2], spring_3D[edge[1]][2], None]
        z_edges += z_coords

        # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                            y=y_edges,
                            z=z_edges,
                            mode='lines',
                            opacity = .5,
                            marker=dict(size=esize),
                                        #,color=ncol,  # color the nodes according to their community
                                        # either green or mageneta
                                        #colorscale=['lightgreen', 'magenta'],
                                        #line=dict(color='black', width=0.5)),
                            hoverinfo='none')

    # create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                            y=y_nodes,
                            z=z_nodes,
                            mode='markers',
                            marker=dict(symbol='circle',
                                        size=nsize,
                                        color=ncol,  # color the nodes according to their community
                                        # either green or mageneta
                                        colorscale=['lightgreen', 'magenta'],
                                        line=dict(color='black', width=0.5)),
                            #                        text=club_labels,
                            hovertext=nnames,
                            hoverinfo='text')


    # we need to set the axis for the plot
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    # also need to create the layout for our plot
    layout = go.Layout(title='SMSB Consulting Group Org Chart',
                    #width=1000,
                    #height=1000,
                    showlegend=False,
                    scene=dict(xaxis=dict(axis),
                                yaxis=dict(axis),
                                zaxis=dict(axis),
                                ),
                    margin=dict(t=100),
                    hovermode='closest')

    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    fig_network = go.Figure(data=data, layout=layout)
    return fig_network
    #py.plot(fig, filename='SMSB Consulting Group Org Chart')

def draw_table(data, x, y, color_by, title):
    chartdf = data.copy(deep=True).groupby([x,color_by]).agg(sum).sort_values(by = [y], ascending = False).reset_index()
    bar_chart = px.bar(chartdf, x=x, y=y, color=color_by, title=title)
    return bar_chart

def draw_heatmap(data,col, title = 'Relative Workload Intensity Heatmap(click to filter downstream)'):
    monthly_agg = data.groupby([col,'Date']).agg(sum).reset_index()
    emp_monthlyavg = monthly_agg.groupby([col]).agg('mean').reset_index()[[col,'Hours']]
    emp_monthlyavg = emp_monthlyavg.rename(columns={"Hours": "Avg_Monthly"})
    monthly_agg = monthly_agg.merge(emp_monthlyavg)
    monthly_agg['Workload'] =  monthly_agg['Hours']/monthly_agg['Avg_Monthly']
    heatdf = monthly_agg[[col,'Date','Workload']]
    heatdf = pd.pivot_table(heatdf, index='Date', columns=col, values='Workload', aggfunc=np.sum)
    fig = px.imshow(heatdf,range_color =[0,3])
    fig.update_layout(xaxis={"title": title, "tickangle": 45})
    fig.update_xaxes(side="top")
    return fig

def draw_heatmap_contributions(data, title = 'Percent of Client Contribution'):
    client_user_agg = data.groupby(['Client','UserId']).agg(sum).reset_index()
    client_agg = client_user_agg.groupby(['Client']).agg(sum).reset_index()[['Client','Hours']]
    client_agg  = client_agg .rename(columns={"Hours": "client_total"})
    client_user_agg = client_user_agg.merge(client_agg)
    client_user_agg['Contribution'] =  client_user_agg['Hours']/client_user_agg['client_total']
    client_user_agg = client_user_agg.sort_values(by = 'client_total', ascending = False)

    heatdf = pd.DataFrame(client_user_agg[['Client','client_total','UserId','Contribution']])
    heatdf = pd.pivot_table(heatdf, index=['Client','client_total'], columns='UserId', values='Contribution', aggfunc=np.sum).sort_values(by = 'client_total', ascending = False)
    heatdf = heatdf.reset_index()
    heatdf = heatdf.drop('client_total',1)
    heatdf = heatdf.set_index('Client')

    fig = px.imshow(heatdf, aspect = 'auto')
    fig.update_layout(xaxis={"title": title, "tickangle": 45})
    fig.update_xaxes(side="top")
    return fig

mydataframe = set_dataframes(jmraw)
date_min =  mydataframe[0]
date_max =  mydataframe[1]
jmselect_clients=  mydataframe[2]
jmselect_users=  mydataframe[3]

fig_network = draw_network(jmraw)
fig_network_proj = draw_proj_network(jmraw, 1)
pareto_allclients = pareto_plot(jmraw, x='Client', y='Hours',title='Time Allocation by Client', show_pct_y=False, pct_format='{0:.0%}')
pareto_acctsbyclient = pareto_plot(jmraw, x='Account Abbr', y='Hours',title='Time Allocation by Account', show_pct_y=False, pct_format='{0:.0%}')
pareto_employ_clients = pareto_plot(jmraw, x='UserId', y='Hours',title='Time Allocation by Employee', show_pct_y=False, pct_format='{0:.0%}')
bar_act_time = draw_table(jmraw,'Date', 'Hours','UserId','Select Account')
bar_client_time = draw_table(jmraw,'Date', 'Hours','UserId','Select Client')
bar_employ_clients = draw_table(jmraw,'Date', 'Hours','Client','Select Employee')

#nexttab
emp_heatmap = draw_heatmap(jmraw,'UserId')
contribution_heatmap = draw_heatmap_contributions(jmraw)

app.layout =html.Div(children=[
        html.Div(children = [
            html.H1('SMSB Resource Allocation Dashboard', 
                id = 'main_title',
                style={'width' : '75%','display':'inline-block'}),
            dcc.DatePickerRange(
                id='my-date-picker-range',
                min_date_allowed= date_min,
                max_date_allowed= date_max,
                initial_visible_month=date_minusyear,
                start_date=date_minusyear,
                end_date=datetime.now(),
                style={'width' : '25%','display':'inline-block'})
            ]),
        dcc.Tabs([
            dcc.Tab(label='Client Breakdown', children=[
                html.Div(children = [
                    dcc.Graph(figure = pareto_allclients, 
                        id = 'pareto_allclients',
                        style={'width' : '50%','display':'inline-block','vertical-align':'top'}),
                    dcc.Graph(figure = bar_client_time, 
                        id = 'bar_client_time', 
                        style={'width' : '50%','display':'inline-block','vertical-align':'top'})
                    ]),

                html.Div(children = [
                    dcc.Graph(figure = pareto_acctsbyclient, 
                        id = 'pareto_acctsbyclient',
                        style={'width' : '50%','display':'inline-block','vertical-align':'top'}),
                    dcc.Graph(figure = bar_act_time, 
                        id = 'bar_act_time',
                        style={'width' : '50%','display':'inline-block','vertical-align':'top'})
                    ]),
                html.Div(children = [
                    dcc.Graph(figure = pareto_employ_clients , 
                        id = 'pareto_employ_clients',
                        style={'width' : '50%','display':'inline-block','vertical-align':'top'}),
                    dcc.Graph(figure = bar_employ_clients, 
                        id = 'bar_employ_clients',
                        style={'width' : '50%','display':'inline-block','vertical-align':'top'})
                    ])
                ], style={'text-align':'center', 'font-size':22}),

            dcc.Tab(label='Employee Breakdown', children=[
                html.Div(children = [
                    dcc.Graph(figure = emp_heatmap, 
                            id = 'emp_heatmap',
                            style={'width' : '100%'}),
                    dcc.Graph(figure = contribution_heatmap, 
                            id = 'contribution_heatmap',
                            style={'width' : '100%','height' : '200%'}),
                    dcc.Graph(figure = fig_network,
                                    id = 'fig_network',
                                    style={'width' : '50%','height' : '200%','display':'inline-block','vertical-align':'top'}),
                    dcc.Graph(figure = fig_network_proj,
                                    id = 'fig_network_proj',
                                    style={'width' : '50%','height' : '200%','display':'inline-block','vertical-align':'top'})
                ], style={'text-align':'center', 'font-size':22})
                ])
            ]),
        dcc.Store(id='intermediate-value'),
        dcc.Store(id='clientfilter-value')
    ])  

@app.callback(
    Output('intermediate-value', 'data'),     
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'))
def clean_data(start_date,end_date):
    jmupdate = jmraw.copy(deep= True)
    if start_date is not None:
        jmupdate= jmupdate[(jmupdate['Date'] >= start_date)]
    if end_date is not None:
        jmupdate = jmupdate[(jmupdate['Date'] <= end_date)]

    return jmupdate.to_json(date_format='iso', orient='split')

@app.callback(
    Output(component_id= 'pareto_allclients', component_property= 'figure'),
    Output(component_id= 'pareto_acctsbyclient', component_property= 'figure'),
    Output(component_id= 'pareto_employ_clients', component_property= 'figure'),
    Output(component_id= 'bar_client_time', component_property= 'figure'),
    Output(component_id= 'bar_act_time', component_property= 'figure'),
    Output(component_id= 'bar_employ_clients', component_property= 'figure'),
    Output(component_id= 'emp_heatmap', component_property= 'figure'),
    Output(component_id= 'contribution_heatmap', component_property= 'figure'),
    Output(component_id= 'fig_network', component_property= 'figure'),
    Output(component_id= 'fig_network_proj', component_property= 'figure'),
    Input('intermediate-value', 'data'))

def update_daterange(data):
    data = pd.read_json(data, orient='split')


    pareto_allclients = pareto_plot(data, x='Client', y='Hours',title='Time Allocation by Client', show_pct_y=False, pct_format='{0:.0%}')
    pareto_acctsbyclient = pareto_plot(data, x='Account Abbr', y='Hours',title='Time Allocation by Account', show_pct_y=False, pct_format='{0:.0%}')
    pareto_employ_clients = pareto_plot(data, x='UserId', y='Hours',title='Time Allocation by Employee', show_pct_y=False, pct_format='{0:.0%}')
    bar_act_time = draw_table(data,'Date', 'Hours','UserId','Select Account')
    bar_client_time = draw_table(data,'Date', 'Hours','UserId','Select Client')
    bar_employ_clients = draw_table(data,'Date', 'Hours','Client','Select Employee')
    emp_heatmap = draw_heatmap(data,'UserId')
    fig_network = draw_network(data)
    fig_network_proj = draw_proj_network(data,1)
    contribution_heatmap = draw_heatmap_contributions(data)

    return pareto_allclients,pareto_acctsbyclient,pareto_employ_clients, bar_client_time, bar_act_time,bar_employ_clients,emp_heatmap,contribution_heatmap, fig_network,fig_network_proj


@app.callback(
    Output('clientfilter-value', 'data'), 
    Output(component_id= 'pareto_acctsbyclient', component_property= 'figure'),
    Output(component_id= 'pareto_employ_clients', component_property= 'figure'),
    Output(component_id= 'bar_client_time', component_property= 'figure'),
    Output(component_id= 'bar_act_time', component_property= 'figure'),
    Input('pareto_allclients', 'clickData'),
    Input('intermediate-value', 'data'))

def update_actpareto(clickData, data):
    data = pd.read_json(data, orient='split')
    clickclient = clickData['points'][0]['x']
    data = data[data['Client']== clickclient]

    bar_client_time = draw_table(data,'Date', 'Hours','UserId',f'{clickclient} Hours Over Time')
    bar_act_time = draw_table(data,'Date', 'Hours','UserId','Account Hours Over Time')
    pareto_acctsbyclient = pareto_plot(data, x='Account Abbr', y='Hours',title=f'{clickclient} Accounts', show_pct_y=False, pct_format='{0:.0%}')
    pareto_employ_clients = pareto_plot(data, x='UserId', y='Hours',title='B', show_pct_y=False, pct_format='{0:.0%}')

    pareto_acctsbyclient = pareto_plot(data, x='Account Abbr', y='Hours',title=f'{clickclient} Time Allocation by Account', show_pct_y=False, pct_format='{0:.0%}')
    pareto_employ_clients = pareto_plot(data, x='UserId', y='Hours',title=f'{clickclient} Time Allocation by Employee', show_pct_y=False, pct_format='{0:.0%}')
    bar_act_time = draw_table(data,'Date', 'Hours','UserId',f'{clickclient} Select Account to Filter')
    bar_employ_clients = draw_table(data,'Date', 'Hours','Client','Select Employee to Filter')
    
    return data.to_json(date_format='iso', orient='split'), pareto_acctsbyclient, pareto_employ_clients, bar_client_time, bar_act_time

@app.callback(
    Output(component_id= 'bar_act_time', component_property= 'figure'),
    Input('pareto_acctsbyclient', 'clickData'),
    Input('clientfilter-value', 'data'))

def update_acts(clickData, data):
    data = pd.read_json(data, orient='split')
    clickact = clickData['points'][0]['x']
    data = data[data['Account Abbr']== clickact]
    bar_act_time = draw_table(data,'Date', 'Hours','UserId',f'{clickact} Hours Over Time')
    return bar_act_time

@app.callback(
    Output(component_id= 'bar_employ_clients', component_property= 'figure'),
    Input('pareto_employ_clients', 'clickData'),
    Input('intermediate-value', 'data'))

def update_employee(clickData, data):
    data = pd.read_json(data, orient='split')
    clickemp = clickData['points'][0]['x']
    data = data[data['UserId']== clickemp]

    bar_employ_clients = draw_table(data,'Date', 'Hours','Client',f'{clickemp} Hours Over Time')
    return bar_employ_clients

@app.callback(
    Output(component_id= 'contribution_heatmap', component_property= 'figure'),
    Output(component_id= 'fig_network', component_property= 'figure'),
    Output(component_id= 'fig_network_proj', component_property= 'figure'),
    Input('emp_heatmap', 'clickData'),
    Input('intermediate-value', 'data'))

def update_empcharts(clickData, data):
    data = pd.read_json(data, orient='split').drop('level_0',axis =1)
    clickemp = clickData['points'][0]['x']
    client_reach = data[data['UserId']== clickemp]
    data = data[data['Client'].isin(list(client_reach['Client']))].reset_index()
    contribution_heatmap = draw_heatmap_contributions(data, f'Percent of Client Contribution(across {clickemp}s clients)')
    fig_network = draw_network(data)
    fig_network_proj = draw_proj_network(data,1)
    return contribution_heatmap,fig_network,fig_network_proj

if __name__ == '__main__':
    app.run_server(debug=False,dev_tools_ui=False,dev_tools_props_check=False)