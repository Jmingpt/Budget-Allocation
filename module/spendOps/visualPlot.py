import plotly.graph_objects as go


def modelPlot(x, x2, y1, y2, metrics):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x,
                             y=y1,
                             mode='markers',
                             name=f'Cost vs. {metrics}'))

    fig.add_trace(go.Scatter(x=x2,
                             y=y2,
                             mode='lines',
                             name='Trend Line'))

    fig.update_layout(title=f"Spending Optimisation",
                      width=800, height=700,
                      xaxis_title="Spending",
                      yaxis_title=metrics)

    return fig
