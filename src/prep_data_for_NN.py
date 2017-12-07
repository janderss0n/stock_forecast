import pandas as pd

df = pd.read_csv('../data/INVE-B-2013-09-22-2017-09-22.csv', delimiter=';')
df.loc[:, ['Date']] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
df.loc[:, ['Closing price']] = pd.to_numeric(df['Closing price'].str.replace(',','.'))
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df = df.rename(columns={'Closing price':'closing_price'})

df.loc[:, ['year', 'month', 'day', 'closing_price']].to_csv('../data/INVB.csv',
    sep=',', decimal='.', index=False)
