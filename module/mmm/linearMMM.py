import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from .mmmTransform import row_to_pivot
from .visualPlot import modelPlot


def mmm_model(df):
    if df is not None:
        date_range, mmm_df = row_to_pivot(df)
        X = mmm_df.drop('Revenue', axis=1)
        y = mmm_df['Revenue']

        model = LinearRegression()
        model.fit(X, y)
        score = model.score(X, y)

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
        dimension = st.radio('Metrics:', ['Coefficient', 'Contribution', 'ROAS'], horizontal=True)

        weights = pd.Series(
            model.coef_,
            index=X.columns
        )
        base = model.intercept_
        unadj_contributions = X.mul(weights).assign(Base=base)
        base_col = unadj_contributions.pop('Base')
        unadj_contributions.insert(0, 'Base', base_col)
        adj_contributions = (unadj_contributions
                             .div(unadj_contributions.sum(axis=1), axis=0)
                             .mul(y, axis=0)
                             )  # contains all contributions for each week
        sales_from_channel = adj_contributions[X.columns].sum()
        spendings_on_channel = X.sum()
        predicted_roas = sales_from_channel / spendings_on_channel
        predicted_roas = predicted_roas.reset_index()
        predicted_roas.columns = ['params', 'roas']
        plot_df = pd.merge(plot_df, predicted_roas, on='params', how='left')
        fig = modelPlot(plot_df, date_range, dimension)
        st.plotly_chart(fig, use_container_width=True)

        fig, ax = plt.subplots()
        adj_contributions.plot.area(
            figsize=(16, 8),
            linewidth=1,
            title='Contribution Plot of each Channel',
            ylabel='Contribution',
            xlabel='Date',
            ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles[::-1], labels[::-1],
            title='Channels', loc="center left",
            bbox_to_anchor=(1.01, 0.5)
        )
        st.pyplot(fig)
