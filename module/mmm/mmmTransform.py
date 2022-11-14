import pandas as pd
import numpy as np
import holidays


def yearweek(d):
    # d = d.to_pydatetime()
    year = d.isocalendar()[0]
    week = d.isocalendar()[1]
    yw = f'{year}{week:02d}'
    return yw


def get_year(d):
    year = d.isocalendar()[0]
    return year


def row_to_pivot(df):
    date_range = '{} - {}'.format(df['Date'].min().strftime('%Y/%m/%d'), df['Date'].max().strftime('%Y/%m/%d'))
    pivot_tb = pd.pivot_table(df, values='Cost', index=['Date'], columns=['Channel'], aggfunc=np.sum)
    pivot_df = pivot_tb.reset_index().sort_values('Date', ascending=True).reset_index(drop=True)
    pivot_df = pivot_df.fillna(0)

    yr = list(set(pivot_df['Date'].apply(get_year)))
    holiday = pd.DataFrame(holidays.country_holidays('MY', years=yr).keys(), columns=['Date'])
    holiday['Date'] = pd.to_datetime(holiday['Date'])
    holiday['Seasonality'] = 1
    pivot_df = pd.merge(pivot_df, holiday, on='Date', how='left')
    pivot_df['Seasonality'] = pivot_df['Seasonality'].fillna(0)

    revenue_df = df.groupby('Date') \
                   .agg({'Revenue': np.sum}) \
                   .reset_index() \
                   .sort_values('Date', ascending=True) \
                   .reset_index(drop=True)
    mmm = pd.merge(pivot_df, revenue_df, on='Date', how='left')
    mmm['YearWeek'] = mmm['Date'].apply(yearweek)
    mmm_df = mmm.groupby(['YearWeek']).agg(np.sum)
    mmm_df['Seasonality'] = [season if season == 0 else 1 for season in mmm_df['Seasonality']]
    # mmm_df = mmm.set_index('Date')

    return date_range, mmm_df
