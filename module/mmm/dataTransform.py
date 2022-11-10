import streamlit as st
import pandas as pd
from io import StringIO
from .mapFBGA import dataMapping


def readData(file_connection_method):
    if file_connection_method == 'Upload from local.':
        datasource_type = st.radio('Select data source:', ['Google Analytics', 'Google Analytics and Facebook Ads'], horizontal=True)

        if datasource_type == 'Google Analytics and Facebook Ads':
            uploaded_files = st.file_uploader('Upload your files', accept_multiple_files=True, type=['csv'])

            if uploaded_files:
                for f in uploaded_files:
                    if 'ga' in str(f.name).lower() or 'google' in str(f.name).lower():
                        bytes_data = f.read()
                        s = str(bytes_data, 'utf-8')
                        data = StringIO(s)
                        df_ga = pd.read_csv(data)
                        df_ga['Date'] = pd.to_datetime(df_ga['Date'])

                    elif 'fb' in str(f.name).lower() or 'facebook' in str(f.name).lower():
                        bytes_data = f.read()
                        s = str(bytes_data, 'utf-8')
                        data = StringIO(s)
                        df_fb = pd.read_csv(data)
                        df_fb['Date'] = pd.to_datetime(df_fb['Date'])

                df = dataMapping(df_ga, df_fb)
                return df

        elif datasource_type == 'Google Analytics':
            uploaded_file = st.file_uploader('Upload your files', accept_multiple_files=False, type=['csv'])

            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df[df['Channel'] != 'Facebook'].reset_index(drop=True)
                return df

    elif file_connection_method == 'Connect to BigQuery.':
        return st.write('Developing')
