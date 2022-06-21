
################################
import pathlib
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, ALL

import plotly.express as px

import pandas as pd
import numpy as np
from pickle import load

import sklearn
import openpyxl

import base64
import io

######################################

####app
app = dash.Dash(external_stylesheets=[dbc.themes.MORPH])  # MORPH   SOLAR
server = app.server

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

# UE dataset
df_UE = pd.read_csv(DATA_PATH.joinpath("UE.csv"))

###########For the Introduction#######
run_info = "To calculate your data center PUE& WUE, you need to choose (1) the climate zone your data center locates in, (2) the size of your data center, (3) the cooling system your data center uses," + \
           " and (4) your data center setpoints and data center system variables (see reference for variable definition), then (5) click the Run button."
climate_info = "The user can select a climate zone, and the model will use the default climate dataset for that climate zone to calculate the annual average PUE and WUE value of the data center. Also, users can choose to upload their own climate dataset, preferably in .csv format, which should include 1) dry bulb temperature in degrees Celsius, 2) relative humidity (%), and 3) atmospheric pressure in Pasar, arranged like:"
reference_info = "This online PUE& WUE calculator is a reduced-order model based on our publications in Climate-and technology-specific PUE and WUE estimations for US data centers using a hybrid statistical and thermodynamics-based approach (https://doi.org/10.1016/j.resconrec.2022.106323)" + \
                 " and Statistical analysis for predicting location-specific data center PUE and its improvement potential (https://doi.org/10.1016/j.energy.2020.117556)."
contact_info = "Should you have any questions, please contact Nuoa Lei (nuoalei@lbl.gov) or Eric Masanet (emasanet@ucsb.edu)."
copyright_info = "Copyright (c) 2022 by Nuoa Lei. All rights reserved."
############

# layout components: see layout below
controls = dbc.Card(
    [
        # Get started button: see toggle_modal callback
        html.Div(
            [
                dbc.Button("Get started with click !", id="get started", n_clicks=0),
                html.Hr(),
                dbc.Modal(
                    [
                        dbc.ModalHeader(html.H2("Model Introduction", style={'font-weight': 'bold'})),
                        dbc.ModalBody(
                            [
                                # how to use
                                html.H3("Run the Model", style={'font-weight': 'bold'}),
                                html.Div(run_info),
                                html.Hr(),
                                # how to use
                                html.H3("Climate Inputs", style={'font-weight': 'bold'}),
                                html.Div(climate_info),
                                html.Img(src=app.get_asset_url("df_example.png")),
                                html.Hr(),
                                # reference
                                html.H3("Reference", style={'font-weight': 'bold'}),
                                html.Div(reference_info),
                                html.Hr(),
                                # Contact
                                html.H3("Contact Us", style={'font-weight': 'bold'}),
                                html.Div(contact_info),
                                html.Hr(),
                                html.Div(copyright_info),
                            ]
                        ),
                    ],
                    id="modal",
                    size='lg',
                    is_open=False,
                ),
            ]
        ),
        # climate zone
        html.Div(
            [
                dbc.Label("Climate Zone", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id="Climate Zone",
                    options=[
                        {"label": col, "value": col} for col in df_UE['Climate Zone'].unique()
                    ],
                    value="1A",
                ),
            ]
        ),
        # Upload Your Climate Data
        html.Div(
            [
                dbc.Label("or upload your climate data file", style={'font-weight': 'bold'}),
                dcc.Upload(id='upload-data', children=dbc.Button('Upload File', color="info"), multiple=False),
            ]
        ),
        # data center size
        html.Div(
            [
                dbc.Label("Data Center Size", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id="Data Center Size",
                    options=[
                        {"label": col, "value": col} for col in df_UE['Data center size'].unique()
                    ],
                    value="Small",
                ),
            ]
        ),
        # Cooling system
        html.Div(
            [
                dbc.Label("Cooling System", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id="Cooling System",
                    options=[
                        {"label": col, "value": col} for col in df_UE['Cooling system'].unique()
                    ],
                    value="Direct expansion system",
                ),
            ]
        ),
        html.Hr(),
        html.Div('Setpoints & System Variables:', style={'font-weight': 'bold'}),
        html.Hr(),
        # Control Container: based on --- 3. Display Dynamic Components (see below)
        html.Div(id='Control Container'),
        html.Div(dbc.Button("Run", id='Calculate-state', n_clicks=0)),
    ],
    body=True,
)

# layout components: see layout below
controls2 = dbc.Card(
    [
        html.Div(id='result',
                 style={'width': '40%', 'display': 'inline-block', "font-family": "Helvetica", 'font-weight': 'bold',
                        'color': '#67a9cf', 'fontSize': 12}),
        html.Hr(),
        html.Div([dcc.Graph(id="PUE-graph")],
                 style={'width': '85%', 'display': 'inline-block', "font-family": "Helvetica"}),
        html.Div([dcc.Graph(id="WUE-graph")],
                 style={'width': '85%', 'display': 'inline-block', "font-family": "Helvetica"})],
    body=True,
    style={'text-align': 'center'}
)

# layout
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.Img(src=app.get_asset_url("lab.png"), style={'width': '50%'}),
                        width={"size": 3, "offset": 1}),
                dbc.Col(html.H1("DATA CENTER PUE & WUE CALCULATOR", style={'color': '#3182bd'}), width={"size": 7}),
            ],
            align="start",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                dbc.Col(controls2, md=6, width={"size": 6, "offset": 1})
            ],
            align="start",
            justify="center"
        ),
    ],
    fluid=True,
)


#############################################################################################################
# 1. Get started dialog
@app.callback(
    Output("modal", "is_open"),
    [Input("get started", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open


# 2. Filter Cooling System Based on Data Center Size
# def filter_options1(v):
#     """Disable option v"""
#     mp = {'Large-scale': {'Airside economizer& adiabatic cooling','Water-cooled chiller (waterside economizer)'},
#           'Midsize': {'Water-cooled chiller (airside economizer)','Water-cooled chiller (waterside economizer)',
#              'Water-cooled chiller', 'Air-cooled chiller (airside economizer)','Air-cooled chiller'},
#           'Small': {'Water-cooled chiller', 'Air-cooled chiller','Direct expansion system'}}

#     return [
#         {"label": i, "value": i, "disabled": i not in mp[v]} for i in mp[v]]

# app.callback(Output("Cooling System", "options"), [Input("Data Center Size", "value")])(
#     filter_options1
# )

# 3. Filter Data Center Size Based on Cooling System
# def filter_options2(v):
#     """Disable option v"""
#     mp = {'Water-cooled chiller (waterside economizer)': {'Large-scale',
#               'Midsize'},
#              'Airside economizer& adiabatic cooling': {'Large-scale'},
#              'Water-cooled chiller (airside economizer)': {'Midsize'},
#              'Air-cooled chiller (airside economizer)': {'Midsize'},
#              'Water-cooled chiller': {'Midsize', 'Small'},
#              'Air-cooled chiller': {'Midsize', 'Small'},
#              'Direct expansion system': {'Small'}}
#     return [
#         {"label": i, "value": i, "disabled": i not in mp[v]} for i in mp[v]]

# app.callback(Output("Data Center Size", "options"), [Input("Cooling System", "value")])(
#     filter_options2
# )

# 4. Display Dynamic Components (dynamically display Setpoints & System Variables based on Data Center Size and Cooling System)
dynamic_controls = {}
dynamic_controls[('Large-scale', 'Airside economizer& adiabatic cooling')] = {
    "Supply air drybulb (lower bound)": [10, 18], "Supply air drybulb (higher bound)": [27, 35],
    "Supply air dew point (lower bound)": [-12, -9],
    "Supply air dew point (higher bound)": [15, 27], "Supply air relative humidity (lower bound)": [8, 20],
    "Supply air relative humidity (higher bound)": [60, 95],
    "Windage loss water as a percentage of cooling tower flowrate": [0.005 / 100, 0.5 / 100],
    "UPS efficiency": [90 / 100, 99 / 100], "Delta T (supply/return cooling tower water)": [4, 6],
    "Cycles of concentration": [3, 15], "Sensible heat ratio": [0.95, 0.99], "Fan pressure (CRAH)": [300, 1000],
    "Chiller partial load factor": [0.2, 0.8], "COP relative error to regressed value": [-11 / 100, 11 / 100],
    "Power loss percentage in power transformation and distribution system": [0 / 100, 2 / 100],
    "Delta T (supply/return CRAH air)": [13.9, 19.4], "Fan efficiency (CRAH)": [0.65, 0.9]}
dynamic_controls[('Large-scale', 'Water-cooled chiller (waterside economizer)')] = {
    "Windage loss water as a percentage of cooling tower flowrate": [0.005 / 100, 0.5 / 100],
    "Cycles of concentration": [3, 15], "Delta T (supply/return cooling tower water)": [4, 6],
    "UPS efficiency": [90 / 100, 99 / 100], "Sensible heat ratio": [0.95, 0.99],
    "Heat exchanger effectiveness (CRAH cooling coils)": [0.7, 0.9], "Supply air drybulb (higher bound)": [27, 35],
    "Approach temperature (cooling tower)": [2.8, 6.7], "Delta T (supply/return facility system water)": [5, 10],
    "Fan pressure (CRAH)": [300, 700], "Chiller partial load factor": [0.2, 0.8],
    "Power loss percentage in power transformation and distribution system": [0 / 100, 2 / 100],
    "COP relative error to regressed value": [-11 / 100, 11 / 100]}
dynamic_controls[('Midsize', 'Water-cooled chiller (airside economizer)')] = {"UPS efficiency": [0.80, 0.94],
                                                                              "Supply air relative humidity (lower bound)": [
                                                                                  10, 30],
                                                                              "Supply air relative humidity (higher bound)": [
                                                                                  60, 80],
                                                                              "Supply air drybulb (lower bound)": [15,
                                                                                                                   18],
                                                                              "Supply air drybulb (higher bound)": [27,
                                                                                                                    32],
                                                                              "Supply air dew point (lower bound)": [
                                                                                  -12, -9],
                                                                              "Supply air dew point (higher bound)": [
                                                                                  15, 27],
                                                                              "Fan pressure (CRAH)": [400, 1000],
                                                                              "Delta T (supply/return CRAH air)": [5,
                                                                                                                   10],
                                                                              "COP relative error to regressed value": [
                                                                                  -0.40, 0],
                                                                              "Chiller partial load factor": [0.1, 0.5],
                                                                              "Windage loss water as a percentage of cooling tower flowrate": [
                                                                                  0.005 / 100, 0.5 / 100],
                                                                              "Fan efficiency (CRAH)": [0.6, 0.8],
                                                                              "Power loss percentage in power transformation and distribution system": [
                                                                                  2 / 100, 5 / 100],
                                                                              "Lighting power to IT power ratio": [
                                                                                  2 / 100, 5 / 100],
                                                                              "Approach temperature (cooling tower)": [
                                                                                  2.8, 6.7],
                                                                              "Cycles of concentration": [3, 12],
                                                                              "Delta T (supply/return cooling tower water)": [
                                                                                  4, 6],
                                                                              "Sensible heat ratio": [0.95, 0.99]}
dynamic_controls[('Midsize', 'Water-cooled chiller (waterside economizer)')] = {
    "Windage loss water as a percentage of cooling tower flowrate": [0.005 / 100, 0.5 / 100],
    "Cycles of concentration": [3, 12], "UPS efficiency": [0.80, 0.94],
    "Delta T (supply/return cooling tower water)": [4, 6], "Sensible heat ratio": [0.95, 0.99],
    "Power loss percentage in power transformation and distribution system": [2 / 100, 5 / 100],
    "Fan pressure (CRAH)": [400, 900], "Delta T (supply/return CRAH air)": [5, 10],
    "COP relative error to regressed value": [-0.4, 0],
    "Approach temperature (cooling tower)": [2.8, 6.7], "Supply air drybulb (higher bound)": [27, 32],
    "Delta T (supply/return facility system water)": [5, 10],
    "Heat exchanger effectiveness (CRAH cooling coils)": [0.65, 0.9],
    "Fan efficiency (CRAH)": [0.6, 0.8], "Lighting power to IT power ratio": [2 / 100, 5 / 100],
    "Chiller partial load factor": [0.1, 0.5]}
dynamic_controls[('Midsize', 'Water-cooled chiller')] = {
    "Windage loss water as a percentage of cooling tower flowrate": [0.005 / 100, 0.5 / 100],
    "UPS efficiency": [0.80, 0.94], "Cycles of concentration": [3, 12],
    "Delta T (supply/return cooling tower water)": [4, 6], "COP relative error to regressed value": [-0.40, 0],
    "Sensible heat ratio": [0.95, 0.99], "Chiller partial load factor": [0.1, 0.5],
    "Power loss percentage in power transformation and distribution system": [2 / 100, 5 / 100],
    "Lighting power to IT power ratio": [2 / 100, 5 / 100], "Fan pressure (CRAH)": [400, 900],
    "Delta T (supply/return CRAH air)": [5, 10], "Fan efficiency (CRAH)": [0.6, 0.80],
    "Approach temperature (cooling tower)": [2.8, 6.7]}
dynamic_controls[('Midsize', 'Air-cooled chiller (airside economizer)')] = {"Sensible heat ratio": [0.95, 0.99],
                                                                            "Supply air relative humidity (lower bound)": [
                                                                                10, 30],
                                                                            "Supply air relative humidity (higher bound)": [
                                                                                60, 80],
                                                                            "Supply air dew point (lower bound)": [-12,
                                                                                                                   -9],
                                                                            "Supply air dew point (higher bound)": [15,
                                                                                                                    27],
                                                                            "Supply air drybulb (lower bound)": [15,
                                                                                                                 18],
                                                                            "Supply air drybulb (higher bound)": [27,
                                                                                                                  32],
                                                                            "UPS efficiency": [0.80, 0.94],
                                                                            "Chiller partial load factor": [0.1, 0.5],
                                                                            "Fan pressure (CRAH)": [400, 1000],
                                                                            "COP relative error to regressed value": [
                                                                                -0.4, -0.25],
                                                                            "Delta T (supply/return CRAH air)": [5, 10],
                                                                            "Fan efficiency (CRAH)": [0.6, 0.8]}
dynamic_controls[('Midsize', 'Air-cooled chiller')] = {"Sensible heat ratio": [0.95, 0.99],
                                                       "UPS efficiency": [0.80, 0.94],
                                                       "Power loss percentage in power transformation and distribution system": [
                                                           2 / 100, 5 / 100], "Chiller partial load factor": [0.1, 0.5],
                                                       "COP relative error to regressed value": [-0.4, -0.25],
                                                       "Fan pressure (CRAH)": [400, 900],
                                                       "Delta T (supply/return CRAH air)": [5, 10]}
dynamic_controls[('Small', 'Air-cooled chiller')] = {"Sensible heat ratio": [0.95, 0.99],
                                                     "UPS efficiency": [0.77, 0.85],
                                                     "Chiller partial load factor": [0.1, 0.5],
                                                     "COP relative error to regressed value": [-0.45, -0.3],
                                                     "Fan pressure (CRAH)": [400, 900]}
dynamic_controls[('Small', 'Water-cooled chiller')] = {
    "Windage loss water as a percentage of cooling tower flowrate": [0.005 / 100, 0.5 / 100],
    "Cycles of concentration": [3, 12], "COP relative error to regressed value": [-0.6, -0.2],
    "UPS efficiency": [0.77, 0.85], "Delta T (supply/return cooling tower water)": [4, 6],
    "Sensible heat ratio": [0.95, 0.99],
    "Chiller partial load factor": [0.1, 0.5],
    "Power loss percentage in power transformation and distribution system": [2 / 100, 4 / 100],
    "Lighting power to IT power ratio": [2 / 100, 4 / 100], "Fan pressure (CRAH)": [400, 900],
    "Delta T (supply/return CRAH air)": [5, 8], "Fan efficiency (CRAH)": [0.6, 0.75],
    "Approach temperature (cooling tower)": [2.8, 6.7]}
dynamic_controls[('Small', 'Direct expansion system')] = {"UPS efficiency": [0.77, 0.85],
                                                          "COP relative error to regressed value": [-0.4, 0.2],
                                                          "Sensible heat ratio": [0.95, 0.99],
                                                          "Lighting power to IT power ratio": [2 / 100, 4 / 100],
                                                          "Delta T (supply/return CRAH air)": [5, 8],
                                                          "Fan pressure (CRAH)": [400, 600],
                                                          "Power loss percentage in power transformation and distribution system": [
                                                              2 / 100, 4 / 100]}


@app.callback(
    Output('Control Container', 'children'),
    [
        Input("Data Center Size", "value"),
        Input("Cooling System", "value"),
    ],
)
def generate_and_display(dcs, cs):
    try:
        list = []
        for k, (v1, v2) in dynamic_controls[(dcs, cs)].items():
            list.append(dbc.Label(k, style={'color': '#636363'}))
            list.append(
                dcc.Slider(min=v1, max=v2, value=(v1 + v2) * 0.5, id={'type': 'filter-slider', 'index': k}, marks=None,
                           tooltip={"placement": "bottom", "always_visible": False}))
        return [html.Div([i]) for i in list]
    except:
        return [html.Div(
            'At current, this data center size and cooling system type combination is not supported by the PUE& WUE calculator.',
            style={'color': '#636363'})]


# 5. prerequisites of 5: for user to select own climate data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'xlsx' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        return df
    except Exception as e:
        print(e)
        return None


# 5. PUE/WUE graph
@app.callback(
    [
        Output("result", "children"),
        Output("PUE-graph", "figure"),
        Output("WUE-graph", "figure")],
    [Input("Calculate-state", "n_clicks")],
    [State("Climate Zone", "value"),
     State('upload-data', 'contents'),  # upload
     State('upload-data', 'filename'),  # upload
     State("Data Center Size", "value"),
     State("Cooling System", "value"),
     State({'type': 'filter-slider', 'index': ALL}, 'value')]
)
def make_graph(n, cz, contents, filename, dcs, cs, values):
    df = df_UE[['Climate Zone', 'Data center size', 'Cooling system', 'PUE', 'WUE']]
    user_weather = None
    if contents:
        df = df.loc[(df['Data center size'] == dcs) & (df['Cooling system'] == cs)][
            ['PUE', 'WUE']]  # not filter climate zone
        user_weather = parse_contents(contents, filename).copy()
    else:
        df = df.loc[(df['Climate Zone'] == cz) & (df['Data center size'] == dcs) & (df['Cooling system'] == cs)][
            ['PUE', 'WUE']]

    ####### Function for Calculating PUE/WUE
    def Cal_PUE_WUE(cz, dcs, cs, values):
        # get case
        mp = {('Large-scale', 'Airside economizer& adiabatic cooling'): 1,
              ('Large-scale', 'Water-cooled chiller (waterside economizer)'): 2,
              ('Midsize', 'Water-cooled chiller (airside economizer)'): 3,
              ('Midsize', 'Water-cooled chiller (waterside economizer)'): 4,
              ('Midsize', 'Water-cooled chiller'): 5,
              ('Midsize', 'Air-cooled chiller (airside economizer)'): 6,
              ('Midsize', 'Air-cooled chiller'): 7,
              ('Small', 'Water-cooled chiller'): 8,
              ('Small', 'Air-cooled chiller'): 9,
              ('Small', 'Direct expansion system'): 10}
        case = mp[(dcs, cs)]
        # get model
        scaler = load(open(DATA_PATH.joinpath('scaler_case{}_pue.pkl'.format(case)), 'rb'))
        PUE_model = load(open(DATA_PATH.joinpath('mlp_case{}_pue.pkl'.format(case)), 'rb'))
        WUE_model = load(open(DATA_PATH.joinpath('mlp_case{}_wue.pkl'.format(case)), 'rb'))
        # get climate data based on climate zone:
        if user_weather is None:
            weather = pd.read_excel(DATA_PATH.joinpath("climate data.xlsx"), sheet_name=cz)
        else:
            weather = user_weather.copy()
        # prepare feature to be feed into model
        inputs = pd.concat([weather, pd.DataFrame([values] * weather.shape[0])], axis=1)
        # calculate PUE& WUE
        _pue = PUE_model.predict(scaler.transform(inputs))
        _wue = WUE_model.predict(scaler.transform(inputs))
        PUE = np.mean(np.maximum(1, _pue))
        WUE = np.mean(np.maximum(0, _wue))
        return PUE, WUE

    #######################################

    # cal PUE& WUE
    try:
        PUE, WUE = Cal_PUE_WUE(cz, dcs, cs, values)
    except:
        PUE, WUE = None, None  ### default value
    #######

    d = pd.DataFrame()
    d[' '] = ['typical data center range', 'this estimate']
    d['PUE (kWh/kWh)'] = [df.PUE.mean(), PUE]
    d['PUE_e'] = [(df.PUE.max() - df.PUE.min()) / 2, 0]

    d['WUE (L/kWh)'] = [df.WUE.mean(), WUE]
    d['WUE_e'] = [(df.WUE.max() - df.WUE.min()) / 2, 0]

    # plot of PUE
    fig1 = px.scatter(d, x=" ", y='PUE (kWh/kWh)', error_y="PUE_e")
    fig1.update_traces(marker=dict(size=12, color='#feb24c'), line=dict(width=2, color='DarkSlateGrey'))
    fig1.update_layout({'plot_bgcolor': '#f7fbff', 'paper_bgcolor': '#f7fbff'})  ##f7f7f7
    fig1.update_xaxes(tickfont_size=15, ticks="outside", ticklen=5, tickwidth=1)
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#a6bddb')

    # plot of WUE
    fig2 = px.scatter(d, x=" ", y='WUE (L/kWh)', error_y="WUE_e")
    fig2.update_traces(marker=dict(size=12, color='#2ca25f'), line=dict(width=2, color='DarkSlateGrey'))
    fig2.update_layout({'plot_bgcolor': '#f7fbff', 'paper_bgcolor': '#f7fbff'})
    fig2.update_xaxes(tickfont_size=15, ticks="outside", ticklen=5, tickwidth=1)
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#a6bddb')

    # Final text displayed
    line1 = "Typical Data Centers PUE range: {} ~ {} kWh/kWh".format(round(df.PUE.min(), 2), round(df.PUE.max(), 2))
    line2 = "Typical Data Centers WUE range: {} ~ {} L/kWh".format(round(df.WUE.min(), 2), round(df.WUE.max(), 2))
    line3 = "Expected Data Center PUE Value: {} kWh/kWh".format(round(PUE, 2) if PUE else 'nan')
    line4 = "Expected Data Center WUE Value: {} L/kWh".format(round(WUE, 2) if WUE else 'nan')

    line5 = "Calculated based on climate data of Climate Zone {}:".format(cz)
    if filename:
        line5 = "Calculated based on user-uploaded climate data {}:".format(filename)

    res = html.Div([
        # dbc.Spinner(size="sm"),
        # dbc.Progress(label="100%",value=100, color="#a6bddb",style={"height": "15px"}),
        html.Div(dbc.Row(line5, style={'color': '#2b8cbe'})),
        html.Hr(),
        html.Div(dbc.Row(line1)),
        html.Div(dbc.Row(line2)),
        html.Div(dbc.Row(line3)),
        html.Div(dbc.Row(line4)),
    ])
    return res, fig1, fig2


##### Run_server
if __name__ == "__main__":
    app.run_server(debug=False)
