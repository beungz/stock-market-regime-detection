from prj_dataprep import load_data, feature_engineering, prepare_data
from prj_model import GMM_regime_detection, KMeans_regime_detection, AgglomerativeClustering_regime_detection, LSTMAutoencoder_GMM_regime_detection
from prj_model import visualize_clustered_regimes

import warnings



def main():
    '''Prepare data, do feature engineering, train model, then make prediction'''

    warnings.filterwarnings("ignore")

    print('\n1. Download stock trading data from Yahoo Finance')
    startdate = '1999-02-09' # 200 trading days before the first trading day of year 2000. This is to support calculation of some trading indicators that requires 200 historical trading days.
    enddate = '2025-04-18'
    price_df = load_data(startdate, enddate)

    print('\n2. Feature Engineering')
    price_df = feature_engineering(price_df)

    print('\n3. Feature selection and prepare data for model training')
    selected_features = ['EMA_cross', 'Rolling_return_6M', 'Drawdown']
    X_scaled, n_features = prepare_data(price_df, selected_features)

    print('\n4. Gaussian Mixture: Train and hyperparameter tuning \n')
    gmm_best_model, gmm_y_regime_state, gmm_silhouette_score = GMM_regime_detection(X_scaled, 3)
    visualize_clustered_regimes(gmm_y_regime_state, price_df.index, price_df['SET_Close'].values, 'Gaussian Mixture Model', 'GMM_market_regime.png')

    print('\n5. K-Means Clustering: Train and hyperparameter tuning \n')
    KMeans_regime_detectionkmeans_best_model, kmeans_y_regime_state, kmeans_silhouette_score = KMeans_regime_detection(X_scaled, 3)
    visualize_clustered_regimes(kmeans_y_regime_state, price_df.index, price_df['SET_Close'].values, 'K-Means Clustering', 'KMeans_market_regime.png')

    print('\n6. Agglomerative Clustering: Train and hyperparameter tuning \n')
    agglo_best_model, agglo_y_regime_state, agglo_silhouette_score = AgglomerativeClustering_regime_detection(X_scaled, 3)
    visualize_clustered_regimes(agglo_y_regime_state, price_df.index, price_df['SET_Close'].values, 'Agglomerative Clustering', 'Agglo_market_regime.png')

    print('\n7. LSTM Autoencoder + Gaussian Mixture: Train and hyperparameter tuning \n')
    autoencoder_best_model, autoencoder_y_regime_state, autoencoder_silhouette_score = LSTMAutoencoder_GMM_regime_detection(X_scaled, 3, n_features)
    visualize_clustered_regimes(autoencoder_y_regime_state, price_df.index[autoencoder_best_model.window_size:], price_df['SET_Close'].values[autoencoder_best_model.window_size:], 'LSTM Autoencoder + Gaussian Mixture', 'Autoencoder_market_regime.png')
    


if __name__ == '__main__':
	main()