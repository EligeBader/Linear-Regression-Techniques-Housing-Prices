'''
This notebook is designed to demonstrate various data preprocessing and machine learning techniques for predicting housing 
prices. It includes functions for loading, cleaning, encoding, and transforming data, as well as training and evaluating models.
The primary focus is on using linear regression to predict housing prices, with metrics such as mean absolute error and mean 
squared error used to evaluate model performance. 
'''
# Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder, LabelEncoder, OrdinalEncoder, MinMaxScaler, OneHotEncoder, power_transform, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import yeojohnson
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import dill
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input,BatchNormalization, Dropout
from keras.callbacks import EarlyStopping



"""# Load data"""

def load_data(file):

    """
    Load data from a CSV file.

    Parameters:
    file (str): Path to the CSV file.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    df = pd.read_csv(file)

    return df

with open('read_file.pickle', 'wb') as f:
    dill.dump(load_data, f)




"""# Drop features"""

def drop_features(df, features_to_drop=[]):

    """
    Drop specified features from the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    features_to_drop (list): List of column names to drop.

    Returns:
    pd.DataFrame: DataFrame with specified features dropped.
    """

    df = df.drop(columns=features_to_drop)

    return df

with open('drop_features.pickle', 'wb') as f:
    dill.dump(drop_features, f)




"""# Split Data"""

def split_data(df, target,col_dropped, feature_selected= None):
    
    """
    Split the data into training and testing sets.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    target (str): Target column name.
    col_dropped (list): List of columns to drop.
    feature_selected (list, optional): List of selected features. Defaults to None.

    Returns:
    tuple: X_train, X_test, y_train, y_test
    """

    if feature_selected == None:
        X = df.drop(columns= [target] + col_dropped)
        y = df[target]

    else:
        X = df[feature_selected]
        y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


with open('split_data.pickle', 'wb') as f:
    dill.dump(split_data, f)




"""# Clean Data"""

def clean_data(df, train=True, target= [], feature_names=None):

    """
    Clean the data by imputing missing values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    train (bool): Whether the data is for training or not. Defaults to True.
    target (list): List of target columns. Defaults to [].
    feature_names (list, optional): List of feature names. Defaults to None.

    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """

    #Use SimpleImputer
    numeric_imputer_file = 'numeric_imputer.pickle'
    categorical_imputer_file = 'categorical_imputer.pickle'


    # Separate numeric and categorical features
    numeric_features = df.drop(columns=target).select_dtypes(exclude=object).columns.tolist()
    categorical_features = df.select_dtypes(include=object).columns.tolist()


    if train:

        if numeric_features:
            # Impute missing values for numeric features
            numeric_imputer = KNNImputer(n_neighbors = 5)
            numeric_imputer.fit(df[numeric_features])
            df[numeric_features] = numeric_imputer.transform(df[numeric_features])

            with open(numeric_imputer_file, 'wb') as f:
                dill.dump(numeric_imputer,f)

            # print(numeric_features)

        if categorical_features:
            # Impute missing values for categorical features
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            categorical_imputer.fit(df[categorical_features])
            df[categorical_features] = categorical_imputer.transform(df[categorical_features])

            feature_names = df.columns.tolist()

            with open(categorical_imputer_file, 'wb') as f:
                dill.dump(categorical_imputer,f)

    else:

        if os.path.exists('feature_names.pickle'):
            with open('feature_names.pickle', 'rb') as f:
                feature_names = dill.load(f)


            for feature in feature_names:
                if feature not in df.columns:
                    df[feature] = 0

            df = df[feature_names]

        if numeric_features and os.path.exists(numeric_imputer_file):

            with open(numeric_imputer_file, 'rb') as f:
                numeric_imputer = dill.load(f)
            # print(numeric_features)
            # print(numeric_imputer.get_feature_names_out())

            df[numeric_features] = numeric_imputer.transform(df[numeric_features])

            # print(numeric_features)

        if categorical_features and os.path.exists(categorical_imputer_file):

            with open(categorical_imputer_file, 'rb') as f:
                categorical_imputer = dill.load(f)

            df[categorical_features] = categorical_imputer.transform(df[categorical_features])


    return df



with open('clean_data.pickle', 'wb') as f:
    dill.dump(clean_data, f)




"""# Encode Data"""

def encode_data(df, encoding_methods, train, target=[]):

    """
    Encode the data using specified encoding methods.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    encoding_methods (dict): Dictionary specifying encoding methods for columns.
    train (bool): Whether the data is for training or not.
    target (list): List of target columns. Defaults to [].

    Returns:
    pd.DataFrame: Encoded DataFrame.
    """

    file_name = 'encoders.pickle'

    target_cols = [col for col, method in encoding_methods.items() if method == 'target' and col in df.columns]
    ordinal_cols = [col for col, method in encoding_methods.items() if method == 'ordinal' and col in df.columns]
    encoders = {}

    if train:

        if target_cols:
            target_encoder = TargetEncoder(target_type='continuous')
            target_encoder.fit(df[target_cols], df[target])
            encoded_data = target_encoder.transform(df[target_cols])
            for i, col in enumerate(target_cols):
                df[col] = encoded_data[:, i]
            encoders['target'] = target_encoder

        if ordinal_cols:
            ordinal_encoder = OrdinalEncoder()
            ordinal_encoder.fit(df[ordinal_cols])
            df[ordinal_cols] = ordinal_encoder.transform(df[ordinal_cols])
            encoders['ordinal'] = ordinal_encoder

        with open(file_name, 'wb') as f:
            dill.dump(encoders, f)


    else:
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                encoders = dill.load(f)

            print(encoders)
            # Transform the categorical columns
            if 'target' in encoders and target_cols:
                encoded_data = encoders['target'].transform(df[target_cols])
                for i, col in enumerate(target_cols):
                    df[col] = encoded_data[:, i]


            if 'ordinal' in encoders and ordinal_cols:
                df[ordinal_cols] = encoders['ordinal'].transform(df[ordinal_cols])
        else:
            raise FileNotFoundError(f"Encoders file not found: {file_name}")

    return df


with open("encode_data.pickle", 'wb') as f:
    dill.dump(encode_data,f)




"""# Transform Target"""

def transform_data(df, target):

    """
    Transform the target data using PowerTransformer.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    target (str): Target column name.

    Returns:
    pd.DataFrame: DataFrame with transformed target.
    """

    file_name ="powertransformer.pickle"

    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            pt = dill.load(f)

        yj_target = pt.transform(df[target].values.reshape(-1,1))
        df['transform_target'] = yj_target


    else:
        pt = PowerTransformer(method='box-cox')
        pt.fit(df[target].values.reshape(-1,1))

        yj_target = pt.transform(df[target].values.reshape(-1,1))
        df['transform_target'] = yj_target

        with open(file_name, 'wb') as f:
            dill.dump(pt, f)

    return df


with open('transformed_data.pickle', 'wb') as f:
    dill.dump(transform_data, f)




"""# Train Model"""

def train_model(model_class, xtrain, ytrain, param_grid={}, best_combination=False ,  **args):

    """
    Train a model using the specified model class and parameters.

    Parameters:
    model_class (class): Model class to be used for training.
    xtrain (pd.DataFrame): Training features.
    ytrain (pd.Series): Training target.
    param_grid (dict, optional): Parameter grid for GridSearchCV. Defaults to {}.
    best_combination (bool, optional): Whether to use GridSearchCV for best parameter combination. Defaults to False.
    **args: Additional arguments for the model class.

    Returns:
    model: Trained model.
    """

    if best_combination:
        model = model_class(**args)
        
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(xtrain, ytrain)

        best_model = grid_search.best_estimator_
        best_model.fit(xtrain, ytrain)

        model_to_save = best_model

    else:
        model = model_class(**args)
        model.fit(xtrain, ytrain)

        model_to_save = model


    with open('trained_model.pickle', 'wb') as f:
        dill.dump(model_to_save, f)

    return model_to_save


with open('train_model.pickle', 'wb') as f:
    dill.dump(train_model, f)



"""# Predict Model"""

def predict_model(df, model):

    """
    Predict using the trained model and optionally inverse transform the predictions.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    model: Trained model.

    Returns:
    np.ndarray: Predictions.
    """

    file_name = "powertransformer.pickle"

    y_new_pred = model.predict(df)


    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
             pt =  pickle.load(f)


        if hasattr(pt, 'inverse_transform'):
           try:
               if not np.any(np.isnan(y_new_pred)) and np.all(np.isfinite(y_new_pred)):
                y_new_pred = pt.inverse_transform(y_new_pred.reshape(-1, 1)).flatten()

                with open('predicted_model.pickle', 'wb') as f:
                    dill.dump(y_new_pred, f)

                if np.isnan(y_new_pred).any():
                    print("Inverse transform produced NaNs. Returning raw predictions.")
                    y_new_pred = model.predict(df)
           except Exception as e:
                    print(f"Inverse transform failed: {e}")
                    y_new_pred = model.predict(df)
        else:
            print("Loaded transformer does not have the inverse_transform method.")


    print(f"Predictions after inverse transform (if applicable):{y_new_pred}")

    return y_new_pred


with open('predict_model.pickle', 'wb') as f:
    dill.dump(predict_model, f)



"""# Build Neural Network Model"""


def neural_network_model(X, y, loss='mse', metrics='auc', activations='relu', output_activation='linear', widths=[64], num_layers=2, epochs=50, batch_size=32, learning_rate=0.001, validation_split=0.3333):

    """
    Build and train a neural network model.

    Parameters:
    X (pd.DataFrame): Input features.
    y (pd.Series): Target values.
    loss (str, optional): Loss function. Defaults to 'mse'.
    metrics (str, optional): Metrics for evaluation. Defaults to 'auc'.
    activations (str or list, optional): Activation function(s) for hidden layers. Defaults to 'relu'.
    output_activation (str, optional): Activation function for the output layer. Defaults to 'linear'.
    widths (list, optional): List of widths for hidden layers. Defaults to [64].
    num_layers (int, optional): Number of hidden layers. Defaults to 2.
    epochs (int, optional): Number of epochs for training. Defaults to 50.
    batch_size (int, optional): Batch size for training. Defaults to 32.
    learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
    validation_split (float, optional): Fraction of data to use for validation. Defaults to 0.3333.

    Returns:
    tuple: Trained model and training history.
    """
    model_nn = Sequential()
    model_nn.add(Input((X.shape[1],)))
    
    if isinstance(activations, list):
        for i in range(num_layers):
            activation = activations[i % len(activations)]  # Rotate through the activations list
            width = widths[i % len(widths)]  # Rotate through the widths list
            model_nn.add(Dense(width, activation=activation))
            # model_nn.add(Dropout(0.2))
    else:
        for i in range(num_layers):
            width = widths[i % len(widths)]  # Rotate through the widths list
            model_nn.add(Dense(width, activation=activations))
            # model_nn.add(Dropout(0.2))
    
    model_nn.add(Dense(1, activation=output_activation))  # Output layer activation

    #  # Early Stopping callback
    # es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, patience=10)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model_nn.compile(loss=loss, optimizer=opt, metrics=[metrics])

   

    history = model_nn.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    return model_nn, history