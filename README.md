# AIPI 520 Project: Stock Market Regime Detection
### **Author**: Matana Pornluanprasert<br>

This is a python script that train unsupervised learning models, both non-deep learning and deep learning models, to label regime states of Thailand's stock market, with maximum states of 3 (Bullish, Bearish, and Neutral) .<br>

<br>

***
# Modeling Approach<br>
The objective of this model is to label the states of the movement of the stock market, in 3 possible states in each time period (trading day). As there is no official market regime set by the stock market or investment professional, this classificaion is trained without known labels, hence unsupervised. The regime predicted by the model will be 0, 1, or 2, which in most cases, can be assigned as Bullish, Bearish, or Neutral quite easily by observing the trend in the plot between regime and stock market index. Silhouette score is used as an evaluation metrics for hyperparameter tuning. This model is not intended to be used for prediction of future market regime and cannot be used for stock trading.<br>

**Training data**<br>
Historical prices of SET Index (Thailand's stock exchange market index), technical indicators of SET Index, USD Index, US Dow Jones Industrial Average (DJI), and HK Hang Seng Index (HSI), from 2000 to 18 April 2025, including the last 200 trading days in 1999 for calculation of certain indicators.<br>

**Prediction**<br>
Time-series of current/historical stock market regime label (daily, on SET trading days).<br>

**Unsupervised learning model choices**
Non-deep learning:
1. Gaussian Mixture
2. K-Means Clustering
3. Agglomerative Clustering
Deep learning:
4. LSTM Autoencoder + Gaussian Mixture<br>

**Model training and hyperparameter tuning**<br>
In each model training, hyperparameter tuning is done in order to get the best Silhouette score.<br>

**Evaluation**<br>
Silhouette score is used as our evaluation metrics, as it relies only on cluster assignments and distance between samples, without true labels. <br>
However, good clustering in the case of stock market regime detection should mean adjacent time steps stay in the same cluster/regime (implying that regime should be smooth, without changing frequently). Silhouette score may not directly measure how predicted regime time series is smooth or not.<br>
<br>
<br>

***
# Data Preprocessing<br>
Data is fed from Yahoo Finance into price_df dataframe. Forward fill is applied and null values are dropped.<br>
<br>
<br>

***
# Feature Engineering<br>
With closing prices of the four indices, log return are calculated based on 7-day rolling period.
For Thailand's SET Index, the following indicators are calculated using closing index level in all trading days :
- MA50, MA200, EMA20, EMA50, EMA100, EMA200
- MACD, MACD_signal, EMA_ratio, RSI, Return_1m, ROC
- EMA_cross, Drawdown, MA_Slope
- Vol_1m, Rolling_return_6M
<br>
<br>

***
# Feature Selection
Based on correlation matrix, there are 3 groups of features with high correlation among themselves in the same group (>= 0.7):<br>

Group 1: SET_Close, MA50, MA200, EMA20, EMA50, EMA100, EMA200<br>
Group 2: MACD, MACD_signal, EMA_ratio, RSI, Return_1m, SET_LogReturn, ROC<br>
Group 3: EMA_cross, MA_Slope<br>

As most of the created features are highly correlated, due to the fact that most of them are created from the time-series of SET Index closing prices. Only three features are selected to used as training data: <br>
- EMA_cross (from Group 3)
- Rolling_return_6M
- Drawdown

For USD Index, DJI Index, and HSI Index, although thought to be a driver of SET Index, they have quite low correlation with SET Index in the long run, so we do not select these features for model training. <br>
<br>

### **Data scaling**
***
Training data are scaled using StandardScaler for all models<br>
For deep learning model (LSTM Autoencoder + Gaussian Mixture), sliding windows is created on scaled data, with window size of 30 days. The sliding windows is fed into LSTM Autoencoder as training data<br>
<br>
<br>

***

# Requirements and How to run the code

### **Requirements**:<br>
```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.9.2
seaborn==0.13.2
scikit_learn==1.5.1
tensorflow==2.19.0
yfinance==0.2.55
ta==0.11.0
```
<br>

### **How to run the code**:<br>
***
To run the code, type the followings in the terminal<br>

On Windows:<br>

```
py market_regime_detection.py
```

On other systems:<br>

```
python market_regime_detection.py
```

<br>
<br>

# Disclaimer
This model cannot predict future stock index level. It intends to classify current and historical regime states of the market movement.

Nothing in this content constitutes a recommendation to buy, sell, or hold any security, financial product, or instrument.
Always consult with a qualified financial advisor or licensed investment professional before making any investment decisions.
Trading in stocks and other financial markets involves risk of loss, and past performance is not indicative of future results.
The author and publisher of this content are not liable for any losses or damages resulting from the use of this information.

***
