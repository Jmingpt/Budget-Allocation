import streamlit as st
import pandas as pd
import statsmodels.tsa.api as tsa
from sklearn.linear_model import LinearRegression

from .mmmTransform import row_to_pivot
from .visualPlot import modelPlot, adstock_plot, predict_model


def adstock(x, theta):
    return tsa.filters.recursive_filter(x, theta)


def saturation(x, beta):
    return x ** beta


def mmm_model(df):
    if df is not None:
        date_range, mmm_df = row_to_pivot(df)
        X = mmm_df.drop('Revenue', axis=1)
        y = mmm_df['Revenue']
        model = LinearRegression()
        # model = Lasso(alpha=1)
        model.fit(X, y)
        score = model.score(X, y)
        y_pred = model.predict(X)

        coef = []
        for i, j in zip(model.coef_, X.columns):
            coef.append([i,j])
        plot_df = pd.DataFrame(coef, columns=['coef', 'params'])
        plot_df['mean_input'] = X.mean().values
        plot_df['contribution'] = plot_df['coef'] * plot_df['mean_input']

        model_cols = st.columns((1, 1))
        with model_cols[0]:
            st.subheader(f'Base Sales (Weekly): RM {model.intercept_:.2f}')
        with model_cols[1]:
            st.subheader(f"R\u00b2: {score:.4f}")
        dimension = st.radio('Metrics:', ['Coefficient', 'Contribution'], horizontal=True)
        fig = modelPlot(plot_df, date_range, dimension)
        st.plotly_chart(fig, use_container_width=True)

        fig_model = predict_model(X.index, y, y_pred)
        st.plotly_chart(fig_model, use_container_width=True)

        adstock_cols = st.columns((1, 1))
        with adstock_cols[0]:
            st.write('Adstock')
            theta = st.number_input('Theta:', min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            channel1 = st.selectbox('Channel', [c for c in X.columns], key='adstock')
            cha1 = X[channel1]
            fig1 = adstock_plot('Adstock', adstock, cha1, theta)
            st.plotly_chart(fig1, use_container_width=True)

        with adstock_cols[1]:
            st.write('Diminishing Return')
            beta = st.number_input('Beta:', min_value=0.0, max_value=1.0, value=0.1, step=0.05)
            channel2 = st.selectbox('Channel', [c for c in X.columns], key='diminishing')
            cha2 = X[channel2]
            fig2 = adstock_plot('Diminishing Return', saturation, cha2, beta)
            st.plotly_chart(fig2, use_container_width=True)

