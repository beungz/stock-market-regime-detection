import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # To ignore Tensorflow warning "oneDNN custom operations are on..."

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Flatten, RepeatVector, Input, TimeDistributed

import itertools



def visualize_clustered_regimes(regime_states, date_index, market_index_close, model_name, graph_file_name):
    # Visualize clustered regimes, predicted by the model

    plt.figure(figsize=(20, 7))
    plt.plot(date_index, market_index_close, label='SET Market Index')
    plt.fill_between(date_index, market_index_close, where=(regime_states == 2), color='orange', alpha=0.5, label='Market State 2')
    plt.fill_between(date_index, market_index_close, where=(regime_states == 1), color='blue', alpha=0.5, label='Market State 1')
    plt.fill_between(date_index, market_index_close, where=(regime_states == 0), color='grey', alpha=0.5, label='Market State 0')
    plt.title(model_name + ': Market Index with Regime States')
    plt.legend()
    plt.savefig("figures/" + graph_file_name)
    plt.show()




def silhouette_scorer(estimator, X):
    '''For Non-deep learning models: Calculate silhouette score to be used as evaluation metric for GridSearchCV'''
    labels = estimator.fit_predict(X)
    # Return -1 if only one cluster
    if len(set(labels)) == 1:
        return -1
    return silhouette_score(X, labels)



def GMM_regime_detection(X_scaled, n_components):
    '''Training model using Gaussian Mixture, and use GridSearchCV for hyperparameter tuning, 
        then return the best model, predicted market regime labels, together with the best Silhouette score'''
    
    # Initiate GMM model
    model = GaussianMixture(n_components=n_components, max_iter=100000, n_init=30, random_state=100)

    # Define the list of hyperparameters
    param_grid = {
        'covariance_type': ['full', 'diag', 'tied'],
        'init_params': ['kmeans'],
        'reg_covar': [1e-6, 1e-4, 1e-2],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=[(slice(None), slice(None))], scoring=silhouette_scorer, verbose=1, n_jobs=-1, return_train_score=True)

    # Fit the model to the training set
    grid_search.fit(X_scaled)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    silhouette_score = grid_search.best_score_
    print("Best Silhouette Score: {:.6f}".format(silhouette_score))

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    # Return regime state of X_scaled
    y_regime_state = best_model.predict(X_scaled)

    return best_model, y_regime_state, silhouette_score



def KMeans_regime_detection(X_scaled, n_clusters):
    '''Training model using K-Means Clustering, and use GridSearchCV for hyperparameter tuning, 
        then return the best model, predicted market regime labels, together with the best Silhouette score'''
    
    # Initiate KMeans Clustering model
    model = KMeans(n_clusters=n_clusters, max_iter=500, random_state=100)

    # Define the list of hyperparameters
    param_grid = {
        'init': ['k-means++', 'random'],
        'n_init': [10, 20, 30],
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=[(slice(None), slice(None))], scoring=silhouette_scorer, verbose=1, n_jobs=-1, return_train_score=True)

    # Fit the model to the training set
    grid_search.fit(X_scaled)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    silhouette_score = grid_search.best_score_
    print("Best Silhouette Score: {:.6f}".format(silhouette_score))

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    # Return regime state of X_scaled
    y_regime_state = best_model.predict(X_scaled)

    return best_model, y_regime_state, silhouette_score



def AgglomerativeClustering_regime_detection(X_scaled, n_clusters):
    '''Training model using Agglomerative Clustering, and use GridSearchCV for hyperparameter tuning, 
        then return the best model, predicted market regime labels, together with the best Silhouette score'''
    
    # Initiate Agglomerative Clustering model
    model = AgglomerativeClustering(n_clusters=n_clusters)

    # Define the list of hyperparameters
    param_grid = {
        'linkage': ['complete', 'average', 'single'],
        'metric': ['euclidean', 'manhattan', 'cosine'] 
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=[(slice(None), slice(None))], scoring=silhouette_scorer, verbose=1, n_jobs=-1, return_train_score=True)

    # Fit the model to the training set
    grid_search.fit(X_scaled)

    # The optimal parameters
    print("Best parameters found: ", grid_search.best_params_)
    silhouette_score = grid_search.best_score_
    print("Best Silhouette Score: {:.6f}".format(silhouette_score))

    # Evaluate the best model
    best_model = grid_search.best_estimator_

    # Return regime state of X_scaled
    y_regime_state = best_model.fit_predict(X_scaled)

    return best_model, y_regime_state, silhouette_score



class LSTMAutoencoder_GMM():
    '''Train LSTM Autoencoder + GMM to detect stock market regime label'''
    def __init__(self, lstm_units=64, latent_dim=16, batch_size=64, epochs=30, gmm_components=3, init_params='kmeans', covariance_type='full', reg_covar=1e-6, window_size=30, n_features=None):
        self.lstm_units = lstm_units
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.gmm_components = gmm_components
        self.init_params = init_params
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.window_size = window_size
        self.n_features = n_features


    def prep_dataloaders_and_create_sequences(self, X_train):
        '''Create sliding windows for LSTM Autoencoder'''

        X_trainloader = np.array([X_train[i:i+self.window_size] for i in range(len(X_train) - self.window_size)])
        X_trainloader = X_trainloader.reshape((X_trainloader.shape[0], X_trainloader.shape[1], self.n_features))  # Dim: [samples, timesteps, n_features]

        return X_trainloader


    def build_and_train_autoencoder(self, X_trainloader):
        '''Build and train LSTM Autoencoder, from X_trainloader, and return model encoder'''
        tf.keras.backend.clear_session()

        # Encoder
        input_seq = Input(shape=(self.window_size, self.n_features))
        encoded = LSTM(self.lstm_units, return_sequences=False)(input_seq)
        encoded = Dense(self.latent_dim, activation='relu')(encoded)

        # Decoder
        decoded = RepeatVector(self.window_size)(encoded)
        decoded = LSTM(self.lstm_units, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(self.n_features))(decoded)

        # Autoencoder Model
        autoencoder = Model(input_seq, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.summary()

        # Train Autoencoder
        autoencoder.fit(X_trainloader, X_trainloader, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        encoder = Model(inputs=input_seq, outputs=encoded)

        return encoder


    def autoencoder_silhouette_scorer(self, X_train):
        '''For LSTM Autoencoder: Calculate silhouette score to be used as evaluation metric for GridSearchCV'''
        if len(set(self.labels)) < 3:
            score = -1
        else:
            score = silhouette_score(X_train[self.window_size:], self.labels)

        # Print silhouette score for the current combination of parameter
        print(f"Silhouette Score: {score:.4f}")

        return score


    def fit(self, X_train):
        '''Train LSTM Autoencoder with GMM'''

        # Prepare data for loading into LSTM Autoencoder and then train LSTM Autoencoder
        X_trainloader = self.prep_dataloaders_and_create_sequences(X_train)
        self.encoder = self.build_and_train_autoencoder(X_trainloader)

        # Get latent features returned from LSTM Autoencoder
        latent_features = self.encoder.predict(X_trainloader)

        # Train GMM Model with latent features returned from LSTM Autoencoder
        self.gmm = GaussianMixture(
            n_components=self.gmm_components,
            init_params=self.init_params,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=100
        ).fit(latent_features)

        # Get GMM model prediction of market regime label, with latent features returned from LSTM Autoencoder
        self.labels = self.gmm.predict(latent_features)
        
        # Print all parameters
        self.print_param()

        # Calculate Silhouette score
        self.score = self.autoencoder_silhouette_scorer(X_train)

        return self


    def predict(self, X_trainloader):
        '''Predict market regime label with LSTM Autoencoder and GMM model'''
        latent_features = self.encoder.predict(X_trainloader)
        return self.gmm.predict(latent_features)


    def print_param(self):
        '''Print all parameters'''
        print('\nLSTM Autoencoder Parameters: ' 
                'window_size:', self.window_size,
                ', lstm_units:', self.lstm_units,
                ', latent_dim:', self.latent_dim,
                ', batch_size:', self.batch_size,
                ', epochs:', self.epochs,
        )
        print('GMM Parameters: '
                'gmm_components:', self.gmm_components,
                ', init_param:', self.init_params,
                ', covariance_type:', self.covariance_type,
                ', reg_covar:', self.reg_covar
        )
        return



def LSTMAutoencoder_GMM_regime_detection(X_scaled, n_components, n_features):
    '''Training model using LSTM Autoencoder + Gaussian Mixture model, and hyperparameter tuning, 
        then return the best model, predicted market regime labels, together with the best Silhouette score'''
    
    # Define the list of hyperparameters
    param_grid = {
        'window_size' : [30],
        'n_features' : [n_features],
        # Parameter for LSTM Autoencoder
        'lstm_units': [64],
        'latent_dim': [16],
        'batch_size': [32],
        'epochs': [30],
        # Parameter for GaussianMixture
        'gmm_components': [n_components],                  # number of market regimes
        'init_params': ['kmeans'],              
        'covariance_type': ['full', 'diag'],    
        'reg_covar': [1e-6]                    
    }

    keys = param_grid.keys()
    values = param_grid.values()
    param_combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

    trial_results = []
    best_score = 0
    best_params = None


    for i, params in enumerate(param_combos):
        print(f"\n[{i+1}/{len(param_combos)}] Trying params: {params}")
        
        # Instantiate model with current params
        model = LSTMAutoencoder_GMM(**params)
        
        # Fit the model to the training set
        model.fit(X_scaled)
        current_score = model.score

        trial_results.append({
            'params': params,
            'score': current_score
        })

        if current_score > best_score:
            best_score = current_score
            best_params = params
            best_labels = model.labels
            best_model = model

    # Print Silhouette Score for each combination of parameters
    df_results = pd.DataFrame(trial_results)
    df_results.sort_values('score', ascending=False, inplace=True)
    print('\nSilhouette Score for each combination of parameters')
    print(df_results)

    # The optimal parameters and score
    print(f'\nBest Parameter: {best_params}')
    print(f'Best Silhouette Score: {best_score:.4f}')

    return best_model, best_labels, best_score