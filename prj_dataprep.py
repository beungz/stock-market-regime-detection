import numpy as np
import pandas as pd

import yfinance as yf

from ta.momentum import RSIIndicator
from ta.trend import MACD

from sklearn.preprocessing import StandardScaler



def load_data(startdate, enddate):
    '''Download stock trading data from Yahoo Finance, do formating, drop NA, forward fill, and then return a price dataframe'''
    try: 
        # startdate = '1999-02-09' # This is the start date that is 200 trading days before the first trading day of year 2000. This is to support calculation of some trading indicators that requires 200 historical trading days
        # enddate = '2025-04-18'

        # Download stock data

        # Thailand Stock Market Index
        price_df = yf.download('^SET.BK', start=startdate, end=enddate, interval='1d')
        price_df.columns = price_df.columns.get_level_values(0)

        # USD Index
        usd_index_df = yf.download('DX-Y.NYB', start=startdate, end=enddate, interval='1d')['Close']

        # Dow Jones Industrial Average (DJI)
        dji_df = yf.download('^DJI', start=startdate, end=enddate, interval='1d')['Close']

        # Hang Seng Index (HSI) - Hong Kong
        hsi_df = yf.download('^HSI', start=startdate, end=enddate, interval='1d')['Close']

        # Merge all price dataframes to price_df
        price_df = pd.concat([price_df, usd_index_df, dji_df, hsi_df], axis=1)

        # Change column names
        price_df.columns = ['SET_Close', 'SET_High', 'SET_Low', 'SET_Open', 'SET_Volume', 'DXY', 'DJI', 'HSI']

        # Drop NA for any row with NA in column 'SET_Close'
        price_df = price_df.dropna(subset=['SET_Close'])

        # Forward Fill
        price_df = price_df.ffill()

        return price_df
    
    except Exception as err:
        print(f"Error occurred: {err}")



def feature_engineering(price_df):
    '''Feature engineering on price_df, and return price_df with new features'''

    # Calculate Log Return of each index
    rolling_period = 7

    price_df['SET_MA'] = price_df['SET_Close'].rolling(rolling_period).mean()
    price_df['SET_LogReturn'] = np.log(price_df['SET_MA']/price_df['SET_MA'].shift(1))

    price_df['DXY_MA'] = price_df['DXY'].rolling(rolling_period).mean()
    price_df['DXY_LogReturn'] = np.log(price_df['DXY_MA']/price_df['DXY_MA'].shift(1))

    price_df['DJI_MA'] = price_df['DJI'].rolling(rolling_period).mean()
    price_df['DJI_LogReturn'] = np.log(price_df['DJI_MA']/price_df['DJI_MA'].shift(1))

    price_df['HSI_MA'] = price_df['HSI'].rolling(rolling_period).mean()
    price_df['HSI_LogReturn'] = np.log(price_df['HSI_MA']/price_df['HSI_MA'].shift(1))

    # Drop temporary/redundant/unused columns
    price_df = price_df.drop(['SET_High', 'SET_Low', 'SET_Open', 'DXY', 'DJI', 'HSI', 'SET_MA', 'DXY_MA', 'DJI_MA', 'HSI_MA'], axis=1)

    # Calculate various technical indicators
    price_df['Returns'] = price_df['SET_Close'].pct_change()
    
    price_df['Return_1m'] = price_df['SET_Close'].pct_change(21)
    price_df['Vol_1m'] = price_df['Returns'].rolling(21).std()

    price_df['Rolling_return_6M'] = price_df['SET_Close'].pct_change(126)

    price_df['MA50'] = price_df['SET_Close'].rolling(window=50).mean()
    price_df['MA200'] = price_df['SET_Close'].rolling(window=200).mean()

    price_df['EMA20'] = price_df['SET_Close'].ewm(span=20, adjust=False).mean()
    price_df['EMA50'] = price_df['SET_Close'].ewm(span=50, adjust=False).mean()
    price_df['EMA100'] = price_df['SET_Close'].ewm(span=100, adjust=False).mean()
    price_df['EMA200'] = price_df['SET_Close'].ewm(span=200, adjust=False).mean()

    price_df['EMA_ratio'] = price_df['SET_Close'] / price_df['EMA50']

    price_df['EMA_cross'] = price_df['EMA50'] > price_df['EMA200']

    price_df['MA_Slope'] = price_df['MA200'].diff(20)

    price_df['MACD'] = MACD(price_df['SET_Close']).macd()
    price_df['MACD_signal'] = MACD(price_df['SET_Close']).macd_signal()

    price_df['RSI'] = RSIIndicator(price_df['SET_Close'], window=14).rsi()

    price_df['Drawdown'] = (price_df['SET_Close'] - price_df['SET_Close'].rolling(200).max()) / price_df['SET_Close'].rolling(200).max()

    price_df['ROC'] = price_df['SET_Close'].pct_change(10)

    # Drop NA in new features due to historical price required
    price_df = price_df.dropna()

    return price_df



def prepare_data(price_df, selected_features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(price_df[selected_features])
    n_features = len(selected_features)

    return X_scaled, n_features

