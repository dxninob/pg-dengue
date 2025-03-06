import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.unitroot import PhillipsPerron
import ruptures as rpt
import plotly.graph_objects as go



class TimeSeries():

    def __init__(self, df, department):
        self.department = department
        
        self.df = df[df['DEPARTMENT'] == department]
        self.df = self.df.set_index('DATE')
        self.df = self.df.asfreq('D')
        self.df = self.df.sort_index()

        self.variables = ['CASES', 'TEMPERATURE', 'PRESSURE']
        self.num_orders = [0, 1]
        self.orders = [f'order_{order}' for order in self.num_orders]

        self.data = {variable: {} for variable in self.variables}
        for variable in self.variables:
            for order in self.orders:
                if order == 'order_0':
                    self.data[variable][order] = self.df[variable].interpolate(method='linear')
                else:
                    self.data[variable][order] = self.data[variable]['order_0'].diff().dropna()


    def stationarity_info(self):

        self.stationarity = {variable: {} for variable in self.variables}
        for variable in self.variables:
            for order in self.orders:
                self.stationarity[variable][order] = {}
                self.stationarity[variable][order]['adf_stat'], self.stationarity[variable][order]['adf_pval'], *_ = adfuller(self.data[variable][order])
                self.stationarity[variable][order]['kpss_stat'], self.stationarity[variable][order]['kpss_pval'], *_ = kpss(self.data[variable][order])
                self.stationarity[variable][order]['pp_stat'], self.stationarity[variable][order]['pp_pval'] = PhillipsPerron(self.data[variable][order]).stat, PhillipsPerron(self.data[variable][order]).pvalue


    def autocorrelation_info(self, lags=365):

        self.autocorrelation = {variable: {} for variable in self.variables}
        for variable in self.variables:
            for order in self.orders:
                self.autocorrelation[variable][order] = {}
                self.autocorrelation[variable][order]['acf'] = acf(self.data[variable][order], nlags=lags)
                self.autocorrelation[variable][order]['pacf'] = pacf(self.data[variable][order], nlags=lags)


    def decompose_info(self):

        self.decompose = {variable: {} for variable in self.variables}
        for variable in self.variables:
            for order in self.orders:
                self.decompose[variable][order] = seasonal_decompose(self.data[variable][order], model='additive', extrapolate_trend='freq')


    def structural_changes_info(self, penalty=10):

        for variable in self.variables:
                mean = self.data[variable][self.orders[0]].mean()
                self.cumsum[variable] = np.cumsum(self.data[variable][self.orders[0]] - mean)

                algo = rpt.Pelt(model="rbf").fit(self.data[variable][self.orders[0]].values)
                self.change_points[variable] = algo.predict(pen=penalty)

    
    def correlation_info(self):

        combinations = list(combinations_with_replacement(self.variables, 2))

        self.correlation = {var1: {var2: None for var2 in self.variables} for var1 in self.variables}
        for i, j in combinations:
            correlation = self.data[i][self.orders[0]].corr(self.data[j][self.orders[0]])
            self.correlation[i][j] = correlation
            self.correlation[j][i] = correlation

    
    def lagged_correlation_matrix(self, var, var_lag, year, lag=1, method='pearson'):
        new_col = self.df[var_lag].copy().shift(lag)

        new_df = self.df.copy()
        new_df[f"Lag_{lag}"] = new_col

        new_df = new_df[new_df.index.year == year]
        new_df = new_df[[var, f"Lag_{lag}"]].dropna()

        correlation = new_df.corr(method=method)
        return correlation
    

    def plot_series(self, variable, order=0):
        """Plots the time series data for the given variable and order."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=self.data[variable][f'order_{order}'], mode='lines', name=f'{variable} (Order {order})'))
        fig.update_layout(title=f'{variable} Time Series (Order {order})', xaxis_title='Date', yaxis_title=variable)
        return fig


    def plot_acf(self, variable, lags, order):
        """Plots the ACF for the given variable and order."""
        acf_values = self.autocorrelation[variable][f'order_{order}']['acf'][1:]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(acf_values))), y=acf_values, name='ACF', width=1.2))
        
        fig.update_layout(title=f'ACF of {variable} (Order {order})', xaxis_title='Lags', yaxis_title='Correlation')
        return fig


    def plot_pacf(self, variable, lags, order):
        """Plots the PACF for the given variable and order."""
        pacf_values = self.autocorrelation[variable][f'order_{order}']['pacf'][1:]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1, len(pacf_values))), y=pacf_values, name='PACF', width=1.2))
        
        fig.update_layout(title=f'PACF of {variable} (Order {order})', xaxis_title='Lags', yaxis_title='Correlation')
        return fig


    def plot_trend(self, variable, order):
        """Plots the trend component from the decomposition."""
        decomposition = self.decompose[variable][f'order_{order}']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=decomposition.trend, mode='lines', name='Trend'))
        
        fig.update_layout(title=f'Trend of {variable} (Order {order})', xaxis_title='Date', yaxis_title='Trend')
        return fig


    def plot_seasonality(self, variable, order):
        """Plots the seasonal component from the decomposition."""
        decomposition = self.decompose[variable][f'order_{order}']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=decomposition.seasonal, mode='lines', name='Seasonality'))
        
        fig.update_layout(title=f'Seasonality of {variable} (Order {order})', xaxis_title='Date', yaxis_title='Seasonality')
        return fig


    def plot_residuals(self, variable, order):
        """Plots the residual component from the decomposition."""
        decomposition = self.decompose[variable][f'order_{order}']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.df.index, y=decomposition.resid, mode='lines', name='Residuals'))
        
        fig.update_layout(title=f'Residuals of {variable} (Order {order})', xaxis_title='Date', yaxis_title='Residuals')
        return fig


    def plot_correlation(self, variable, target, lags):
        """Plots the correlation heatmap between a variable and a target over different lags for each year."""

        correlations = {}
        years = sorted(self.df.index.year.unique(), reverse=True)
        # Compute correlations for each year and lag
        for year in years:
            correlations[year] = {}
            for lag in sorted(lags, reverse=True):  # Ensure lags are sorted in descending order
                corr = self.lagged_correlation_matrix(
                    var=variable,
                    var_lag=target,
                    year=year,
                    lag=lag,
                    method="pearson"
                )
                correlations[year][lag] = corr.iloc[0, 1]

        # Convert to DataFrame
        correlation_df = pd.DataFrame(correlations).T  # Years as rows, lags as columns
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_df.values,
            x=[f"{lag} (days)" for lag in correlation_df.columns],  # Lag labels
            y=correlation_df.index.astype(str),  # Year labels
            colorscale="RdBu_r",  # Reverse "RdBu" to match Matplotlib
            zmin=-1,  # Set consistent scale for correlation
            zmax=1,
            text=correlation_df.round(2).astype(str).values,  # Display values
            texttemplate="%{text}",  # Ensure values are visible
            hoverongaps=False
        ))

        # Adjust layout
        fig.update_layout(
            title=f"Correlation between {variable} and {target}",
            xaxis_title="Lag (days)",
            yaxis_title="Year",
            width=800,  # Adjust width to avoid stretching
            height=500,  # Adjust height for readability
            font=dict(size=14)
        )

        return fig
