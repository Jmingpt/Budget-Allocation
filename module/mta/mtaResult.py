import streamlit as st
from .mtaAlgorithm import *
from .visualPlot import modelPlot


def mtaModel(file_connection_method):
    if file_connection_method == 'Upload from local.':
        uploaded_file = st.file_uploader('Upload your files', accept_multiple_files=False, type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)

            data, _ = markov_chain(df, no_iteration=10, no_of_simulation=10000, alpha=5)
            fig = modelPlot(data)
            st.plotly_chart(fig, use_container_width=True)
