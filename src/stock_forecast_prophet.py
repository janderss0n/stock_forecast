import pandas as pd
from fbprophet import Prophet
import os
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
    return pd.read_csv(filename, delimiter=';', decimal=',')


def prepare_data_for_prophet(data, column_to_predict):
    return data.loc[:, ['Date', column_to_predict]].rename(columns={'Date':'ds', column_to_predict:'y'})


def train_model_prophet(df):
    model = Prophet(weekly_seasonality=True,
        yearly_seasonality=True)
    model.fit(df.loc[:, ['ds', 'y']])
    return model


def make_prediction(model, dates_to_predict):
    return model.predict(dates_to_predict)

def plot_prophet_prediction(forecast, filename):
    fig = model.plot(forecast)
    fig.savefig('../plots/stock_forecast_plt_{}.png'.format(filename))
    plt.close(fig)
    fig_comp = model.plot_components(forecast)
    fig_comp.savefig('../plots/stock_forecast_plt_comp_{}.png'.format(filename))
    plt.close(fig_comp)


if __name__ == '__main__':
    #filename = 'OMX30-2013-09-22-2017-09-22.csv' #os.getenv('FILENAME')
    #filename = 'INVE-B-2013-09-22-2017-09-22.csv'
    filename = 'HM-B-2000-09-22-2017-09-22.csv'
    column_to_predict = 'Closing price'
    stock_data = load_data('../data/{}'.format(filename))
    stock_data[column_to_predict] = stock_data[column_to_predict].apply(np.log)
    model = train_model_prophet(prepare_data_for_prophet(stock_data, column_to_predict))
    dates_to_predict = model.make_future_dataframe(periods=1095)
    prediction = make_prediction(model, dates_to_predict)
    plot_prophet_prediction(prediction, filename)
