import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .markovTool import *
from .visualPlot import modelPlot


def markovModel(file_connection_method):
    if file_connection_method == 'Upload from local.':
        uploaded_file = st.file_uploader('Upload your files', accept_multiple_files=False, type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            df = df.astype({'conversion': 'int', 'conversion_value': 'float'})
            df = df.sort_values(['cookie', 'time'], ascending=[False, True])
            df['visit_order'] = df.groupby('cookie').cumcount() + 1
            df_paths = df.groupby('cookie')['channel'].aggregate(lambda x: x.unique().tolist()).reset_index()

            df_last_interaction = df.drop_duplicates('cookie', keep='last')[['cookie', 'conversion']]
            df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='cookie')

            df_paths['path'] = transform_pathlist(df_paths)
            df_paths = df_paths[['cookie', 'path']]
            list_of_paths = df_paths['path']
            total_conversions = sum(path.count('Conversion') for path in df_paths['path'].tolist())
            base_conversion_rate = total_conversions / len(list_of_paths)

            trans_states = transition_states(list_of_paths)
            trans_prob = transition_prob(trans_states, list_of_paths)
            trans_matrix = transition_matrix(list_of_paths, trans_prob)

            trans_plot, ax_trans = plt.subplots(figsize=(10,8))
            sns.heatmap(trans_matrix.drop(['Start', 'Null', 'Conversion'], axis=1) \
                        .drop(['Start', 'Null', 'Conversion'], axis=0) \
                        .round(2), cmap="Greens", ax=ax_trans, annot=True, annot_kws={'fontsize': 10})
            plt.title('Typical User Journeys')
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            plot_cols = st.columns((2, 1))
            with plot_cols[0]:
                st.pyplot(trans_plot)

            removal_effects_dict = removal_effects(trans_matrix, base_conversion_rate)
            attributions = markov_chain_allocations(removal_effects_dict, total_conversions)
            df_plot = pd.json_normalize(attributions).T.reset_index()
            df_plot.columns = ['channel', 'conv']
            fig = modelPlot(df_plot)
            st.plotly_chart(fig, use_container_width=True)


        elif file_connection_method == 'Connect to BigQuery.':
            st.header('Developing')
