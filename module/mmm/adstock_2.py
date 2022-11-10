# import streamlit as st
# import theano.tensor as tt
# import pymc3 as pm
# import matplotlib.pyplot as plt
# import arviz as az
# from .mmmTransform import row_to_pivot
#
#
# def saturate(x, a):
#     return 1 - tt.exp(-a * x)
#
#
# def carryover(x, strength, length=21):
#     w = tt.as_tensor_variable(
#         [tt.power(strength, i) for i in range(length)]
#     )
#
#     x_lags = tt.stack(
#         [tt.concatenate([
#             tt.zeros(i),
#             x[:x.shape[0] - i]
#         ]) for i in range(length)]
#     )
#     return tt.dot(w, x_lags)
#
#
# def adstock(df):
#     if df is not None:
#         _, mmm_df = row_to_pivot(df)
#         X = mmm_df.drop('Revenue', axis=1)
#         y = mmm_df['Revenue']
#
#         with pm.Model() as mmm:
#             channel_contributions = []
#
#             for channel in X.columns:
#                 coef = pm.Exponential(f'coef_{channel}', lam=0.0001)
#                 sat = pm.Exponential(f'sat_{channel}', lam=1)
#                 car = pm.Beta(f'car_{channel}', alpha=2, beta=2)
#
#                 channel_data = X[channel].values
#                 channel_contribution = pm.Deterministic(
#                     f'contribution_{channel}',
#                     coef * saturate(
#                         carryover(
#                             channel_data,
#                             car
#                         ),
#                         sat
#                     )
#                 )
#
#                 channel_contributions.append(channel_contribution)
#
#             base = pm.Exponential('base', lam=0.0001)
#             noise = pm.Exponential('noise', lam=0.0001)
#
#             sales = pm.Normal(
#                 'sales',
#                 mu=sum(channel_contributions) + base,
#                 sigma=noise,
#                 observed=y
#             )
#
#             trace = pm.sample(return_inferencedata=True, tune=3000)
#
#             fig, ax = plt.subplots()
#             az.plot_posterior(
#                 trace,
#                 var_names=['contribution'],
#                 filter_vars='like',
#                 ax = ax
#             )
#             st.pyplot(fig)
