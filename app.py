import streamlit as st
from module.mmm.linearMMM import mmm_model
# from module.mmm.adstock_2 import adstock
from module.mmm.dataTransform import readData
from module.minorTool.sideBar import sidebarTool
from module.mta.mtaResult import mtaModel
from module.markov.dataTransform import markovModel
from module.spendOps.spendOptimise import spend_ops


def run():
    st.set_page_config(
        page_title="Data Modelling",
        page_icon="📈",
        layout="wide",  # centered, wide
        initial_sidebar_state="auto"  # auto, expanded, collapsed
        # menu_items={
        #     "Get Help": "https://www.impersuasion.com/",
        #     "Report a bug": "mailto:jiaminglow@impersuasion.com",
        #     "About": "# This is a header. This is an *extremely* cool app!"
        # }
    )
    st.title('Budget Allocation')

    tool_option = sidebarTool()
    file_connection_method = st.radio('Select method:', ['Upload from local.', 'Connect to BigQuery.'], horizontal=True)

    if tool_option == 'MMM Model':
        df = readData(file_connection_method)
        mmm_model(df)
        # adstock(df)

    elif tool_option == 'MTA Model':
        st.write('Reference: https://www.linkedin.com/pulse/multi-channel-attribution-model-python-sheranga-gamwasam')
        mtaModel(file_connection_method)

    elif tool_option == 'Markov Model':
        st.write('Reference: https://towardsdatascience.com/marketing-channel-attribution-with-markov-chains-in-python-part-2-the-complete-walkthrough-733c65b23323#:~:text=Save-,Marketing%20Channel%20Attribution%20with%20Markov%20Chains%20in%20Python%20%E2%80%94%20Part%202,eventually%20convert%20(or%20not).')
        markovModel(file_connection_method)

    elif tool_option == 'Spend Ops':
        spend_ops(file_connection_method)


if __name__ == "__main__":
    run()
