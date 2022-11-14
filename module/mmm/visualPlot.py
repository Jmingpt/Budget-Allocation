import plotly.graph_objects as go


def modelPlot(df, date_range, dimension):
    if dimension == 'Coefficient':
        filter = 'coef'
    elif dimension == 'Contribution':
        filter = 'contribution'
    elif dimension == 'ROAS':
        filter = 'roas'
    df_plot = df.sort_values(filter, ascending=False)
    x = df_plot['params'].values
    y = [round(i, 2) for i in df_plot[filter].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         text=y,
                         textposition='outside'))

    fig.update_layout(title=f"MMM Model [{date_range}]",
                      height=500,
                      yaxis_title=dimension,
                      yaxis_range=[min(y) - abs(max(y)) / 5, max(y) + abs(max(y)) / 5])

    return fig


def predict_model(x, y, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                             y=y,
                             mode='lines',
                             name='Actual'))

    fig.add_trace(go.Scatter(x=x,
                             y=y_pred,
                             mode='lines',
                             name='Predicted'))

    fig.update_layout(title='Prediction',
                      height=400,
                      yaxis_title='Revenua')

    return fig


def adstock_plot(title, func_, x, hyperparams):
    result = func_(x, hyperparams)
    x = result.index
    y = result.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                             y=y,
                             mode='lines'))

    fig.update_layout(title=title,
                      height=400,
                      yaxis_title=title)

    return fig
