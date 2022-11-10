import streamlit as st
import pandas as pd
import numpy as np
from .visualPlot import modelPlot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def spend_ops(file_connection_method):
    if file_connection_method == 'Upload from local.':
        input_cols = st.columns((1, 1, 2))
        with input_cols[0]:
            metrics = st.radio('Metrics:', ['ROAS', 'Awareness to cart'], horizontal=True)
        with input_cols[1]:
            degree = st.number_input('Degree:', min_value=1, max_value=5, value=2)
        uploaded_file = st.file_uploader('Upload your files', accept_multiple_files=False, type=['csv'])
        if metrics == 'ROAS':
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.fillna(0)
                df = df.sort_values('Date', ascending=True)
                df['year'] = [yr.isocalendar()[0] for yr in df['Date']]
                df['week'] = [week.isocalendar()[1] for week in df['Date']]

                df_plot = df.groupby(['year', 'week']) \
                            .agg({'Cost': np.sum, 'Revenue': np.sum}) \
                            .reset_index()
                df_plot = df_plot[df_plot['Cost'] > 0]
                df_plot = df_plot.sort_values('Cost', ascending=True)
                df_plot['ROAS'] = df_plot['Revenue']/df_plot['Cost']
                x = df_plot['Cost'].values
                x_plot = np.linspace(min(x), max(x), 500)
                y = df_plot['ROAS'].values

                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(x.reshape(-1, 1))
                x_ploy = poly.fit_transform(x_plot.reshape(-1, 1))
                poly_reg_model = LinearRegression()
                poly_reg_model.fit(poly_features, y)
                y_predicted = poly_reg_model.predict(x_ploy)

                fig = modelPlot(x, x_plot, y, y_predicted, metrics)
                st.plotly_chart(fig, use_container_width=True)

        elif metrics == 'Awareness to cart':
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.fillna(0)
                df = df.sort_values('Cost', ascending=True)
                df = df[df['Cost'] > 0]

                x = df['Cost'].values
                x_plot = np.linspace(min(x), max(x), 500)
                y = df['Products ATC'].values

                poly = PolynomialFeatures(degree=degree, include_bias=False)
                poly_features = poly.fit_transform(x.reshape(-1, 1))
                x_ploy = poly.fit_transform(x_plot.reshape(-1, 1))
                poly_reg_model = LinearRegression()
                poly_reg_model.fit(poly_features, y)
                y_predicted = poly_reg_model.predict(x_ploy)

                fig = modelPlot(x, x_plot, y, y_predicted, metrics)
                st.plotly_chart(fig, use_container_width=True)
