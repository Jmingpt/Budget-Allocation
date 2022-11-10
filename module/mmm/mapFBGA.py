import pandas as pd
import numpy as np


def dataMapping(df_ga, df_fb):
    df_ga_tmp = df_ga[df_ga['Channel'] != 'Facebook'].groupby(['Date', 'Channel']) \
                                                     .agg({'Cost': np.sum, 'Revenue': np.sum}) \
                                                     .reset_index()
    df_ga_fb = df_ga[df_ga['Channel'] == 'Facebook'].groupby('Date') \
                                                    .agg({'Revenue': np.sum}) \
                                                    .reset_index()
    df_fb_tmp = df_fb.groupby('Date') \
                     .agg({'Cost': np.sum}) \
                     .reset_index()
    df_fb_tmp = pd.merge(df_fb_tmp, df_ga_fb, on='Date', how='left')
    df_fb_tmp['Channel'] = 'Facebook'
    df_fb_tmp = df_fb_tmp[['Date', 'Channel', 'Cost', 'Revenue']]
    df = pd.concat([df_ga_tmp, df_fb_tmp], ignore_index=True).sort_values('Date', ignore_index=True)
    return df
