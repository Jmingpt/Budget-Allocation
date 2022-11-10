import plotly.graph_objects as go


def modelPlot(df):
    df_plot = df.sort_values('conv', ascending=False)
    x = df_plot['channel'].values
    y = [round(i, 2) for i in df_plot['conv'].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         text=y,
                         textposition='outside'))

    fig.update_layout(title=f"Markov Model",height=500,
                      yaxis_title="Conversions",
                      #yaxis_range=[min(y) - abs(max(y)) / 5, max(y) + abs(max(y)) / 5]
                      )

    return fig