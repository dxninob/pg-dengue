import os
import sys
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import ClassTimeSeries


# General data
variables = ['CASES', 'TEMPERATURE', 'PRESSURE']
departments = ['VALLE', 'ANTIOQUIA', 'ATLANTICO', 'HUILA', 'BOLIVAR']

# Dashboard App
app = Dash(__name__)

# Dashboard layout
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Label("Select Department"),
                dcc.Dropdown(id='dept-dropdown', options=[{'label': d, 'value': d} for d in departments], value='VALLE'),
            ]),

            html.Div([
                html.Label("Select Variable"),
                dcc.Dropdown(id='var-dropdown', options=[{'label': v, 'value': v} for v in variables], value='CASES'),
            ]),

            html.Div([
                html.Label("Enter Lags"),
                dcc.Input(id='lag-input', type='text', value='0,1,7,14,30,365', debounce=True),
            ]),
        ], className="input-container"),
    ], className="fixed-panel"),

    # First 6 rows (2 columns each)
    html.Div([
        html.Div(dcc.Graph(id='series-order-0-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='series-order-1-plot'), className="graph-box"),
    ], className="graph-row"),

    html.Div([
        html.Div(dcc.Graph(id='acf-order-0-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='acf-order-1-plot'), className="graph-box"),
    ], className="graph-row"),

    html.Div([
        html.Div(dcc.Graph(id='pacf-order-0-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='pacf-order-1-plot'), className="graph-box"),
    ], className="graph-row"),

    html.Div([
        html.Div(dcc.Graph(id='trend-order-0-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='trend-order-1-plot'), className="graph-box"),
    ], className="graph-row"),

    html.Div([
        html.Div(dcc.Graph(id='seasonality-order-0-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='seasonality-order-1-plot'), className="graph-box"),
    ], className="graph-row"),

    html.Div([
        html.Div(dcc.Graph(id='residuals-order-0-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='residuals-order-1-plot'), className="graph-box"),
    ], className="graph-row"),

    html.Div([
        html.Div(dcc.Graph(id='correlation-itself-plot'), className="graph-box"),
        html.Div(dcc.Graph(id='correlation-cases-plot'), className="graph-box"),
    ], className="graph-row"),

], className="container")


# Callback to update plots
@app.callback(
    [
        Output('series-order-0-plot', 'figure'),
        Output('series-order-1-plot', 'figure'),
        Output('acf-order-0-plot', 'figure'),
        Output('acf-order-1-plot', 'figure'),
        Output('pacf-order-0-plot', 'figure'),
        Output('pacf-order-1-plot', 'figure'),
        Output('trend-order-0-plot', 'figure'),
        Output('trend-order-1-plot', 'figure'),
        Output('seasonality-order-0-plot', 'figure'),
        Output('seasonality-order-1-plot', 'figure'),
        Output('residuals-order-0-plot', 'figure'),
        Output('residuals-order-1-plot', 'figure'),
        Output('correlation-itself-plot', 'figure'),
        Output('correlation-cases-plot', 'figure')
    ],
    [
        Input('dept-dropdown', 'value'),
        Input('var-dropdown', 'value'),
        Input('lag-input', 'value')
    ]
)


def update_plots(department, variable, lag_input):

    # Convert input string to a list of integers
    try:
        lags = [int(x.strip()) for x in lag_input.split(',') if x.strip().isdigit()]
    except ValueError:
        lags = [0, 1, 7, 14, 30, 365]  # Default if input is invalid

    # Retrieve the TimeSeries object
    ts_obj = info_departments.get(department)

    series_order_0_plot = ts_obj.plot_series(variable, order=0)
    series_order_1_plot = ts_obj.plot_series(variable, order=1)
    acf_order_0_plot = ts_obj.plot_acf(variable, lags, order=0)
    acf_order_1_plot = ts_obj.plot_acf(variable, lags, order=1)
    pacf_order_0_plot = ts_obj.plot_pacf(variable, lags, order=0)
    pacf_order_1_plot = ts_obj.plot_pacf(variable, lags, order=1)
    trend_order_0_plot = ts_obj.plot_trend(variable, order=0)
    trend_order_1_plot = ts_obj.plot_trend(variable, order=1)
    seasonality_order_0_plot = ts_obj.plot_seasonality(variable, order=0)
    seasonality_order_1_plot = ts_obj.plot_seasonality(variable, order=1)
    residuals_order_0_plot = ts_obj.plot_residuals(variable, order=0)
    residuals_order_1_plot = ts_obj.plot_residuals(variable, order=1)

    correlation_itself_plot = ts_obj.plot_correlation(variable, target=variable, lags=lags)
    correlation_cases_plot = ts_obj.plot_correlation(variable, target='CASES', lags=lags)

    return [
        series_order_0_plot,
        series_order_1_plot,
        acf_order_0_plot,
        acf_order_1_plot,
        pacf_order_0_plot,
        pacf_order_1_plot,
        trend_order_0_plot,
        trend_order_1_plot,
        seasonality_order_0_plot,
        seasonality_order_1_plot,
        residuals_order_0_plot,
        residuals_order_1_plot,
        correlation_itself_plot,
        correlation_cases_plot
    ]



if __name__ == "__main__":
    # Read data
    df_merged = pd.read_pickle(f"..\data\stage\data.pkl")

    # Store TimeSeries objects
    global info_departments
    info_departments = {}
    for department in departments:
        obj = ClassTimeSeries.TimeSeries(df=df_merged, department=department)
        obj.stationarity_info()
        obj.autocorrelation_info()
        obj.decompose_info()
        obj.correlation_info()
        info_departments[department] = obj
    
    # Run dashoard app
    app.run_server(debug=True)