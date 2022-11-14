import plotly.graph_objects as go


def modelPlot(df):
    df_plot = df.sort_values('ass_conversion', ascending=False)
    x = df_plot['channel'].values
    y = [round(i, 2) for i in df_plot['ass_conversion'].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x,
                         y=y,
                         text=y,
                         textposition='outside'))

    fig.update_layout(title=f"MTA Model",
                      width=800, height=700,
                      yaxis_title="Assisted Conversion",
                      yaxis_range=[min(y) - abs(max(y)) / 5, max(y) + abs(max(y)) / 5])

    return fig
