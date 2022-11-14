import streamlit as st
import aesara.tensor as at
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
from .mmmTransform import row_to_pivot


def saturate(x, a):
    return 1 - at.exp(-a * x)


def carryover(x, strength, length=21):
    w = at.as_tensor_variable(
        [at.power(strength, i) for i in range(length)]
    )

    x_lags = at.stack(
        [at.concatenate([
            at.zeros(i),
            x[:x.shape[0] - i]
        ]) for i in range(length)]
    )
    return at.dot(w, x_lags)


def adstock(df):
    if df is not None:
        _, mmm_df = row_to_pivot(df)
        X = mmm_df.drop('Revenue', axis=1)
        y = mmm_df['Revenue']

        with pm.Model() as mmm:
            channel_contributions = []

            for channel in X.columns:
                coef = pm.Exponential(f'coef_{channel.lower().replace(" ", "_")}', lam=0.0001)
                sat = pm.Exponential(f'sat_{channel.lower().replace(" ", "_")}', lam=1)
                car = pm.Beta(f'car_{channel.lower().replace(" ", "_")}', alpha=2, beta=2)

                channel_data = X[channel].values
                channel_contribution = pm.Deterministic(
                    f'contribution_{channel.lower().replace(" ", "_")}',
                    coef * saturate(
                        carryover(
                            channel_data,
                            car
                        ),
                        sat
                    )
                )

                channel_contributions.append(channel_contribution)

            base = pm.Exponential('base', lam=0.0001)
            noise = pm.Exponential('noise', lam=0.0001)

            sales = pm.Normal(
                'sales',
                mu=sum(channel_contributions) + base,
                sigma=noise,
                observed=y
            )

            trace = pm.sample(return_inferencedata=True, tune=100)

        fig, ax = plt.subplots()
        az.plot_posterior(
            trace,
            var_names=['contribution'],
            filter_vars='like',
            ax = ax
        )
        st.pyplot(fig)
