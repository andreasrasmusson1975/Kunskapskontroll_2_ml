"""
A module for loading and preprocessing the MNIST dataset with elastic augmentation.

This module provides a single function fetch, which retrieves the MNIST dataset from OpenML,
applies elastic transformations to augment the training data, and saves the processed datasets to disk for quick subsequent loading.

Functions
---------
fetch(from_disk=False)
    Loads MNIST data, applies augmentation, splits the data, and optionally caches results.

Examples
--------
>>> from mnist_loader import fetch
>>> X_train_val, X_train,X_val,X_test,X,y_train_val,y_train,y_val,y_test,y = fetch()
"""

import numpy as np
from elastic_transformer import ElasticTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator, TransformerMixin

def fetch(from_disk: bool = False) -> tuple:
    """
    Load and preprocess the MNIST dataset with optional disk caching.

    Fetches the MNIST dataset, applies elastic transformations to augment the training dataset,
    and shuffles all datasets. Processed datasets are optionally saved as npy files.

    Parameters
    ----------
    from_disk : bool, default=False
        If True, load the processed datasets directly from cached npy files.

    Returns
    -------
    tuple
        
    """
    if not from_disk:
        n_samples = 70000
        t_size = 10000
        # Load from openml
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False,parser='auto')
        X = mnist["data"][:n_samples,:]
        y = mnist["target"].astype(np.uint8)[:n_samples]

        # Split into train, validation and test data
        X_train_val,X_test,y_train_val,y_test = train_test_split(X,y,test_size=t_size)
        X_train,X_val, y_train,y_val = train_test_split(X_train_val,y_train_val,test_size=t_size)

        # Apply elastic transformations to the training data
        et = ElasticTransformer()
        X_train_val = et.fit_transform(X_train_val)
        y_train_val = np.concatenate([y_train_val,y_train_val],axis=0)
        X_train = et.fit_transform(X_train)
        y_train = np.concatenate([y_train,y_train],axis=0)
        X=et.transform(X)
        y=np.concatenate([y,y],axis=0)

        # Shuffle the train_val data
        rng = np.random.default_rng()
        permutations = rng.permutation(X_train_val.shape[0])
        X_train_val=X_train_val[permutations,:]
        y_train_val=y_train_val[permutations]

        # Shuffle the train data
        permutations = rng.permutation(X_train.shape[0])
        X_train=X_train[permutations,:]
        y_train=y_train[permutations]

        # Shuffle all data
        permutations = rng.permutation(X.shape[0])
        X=X[permutations,:]
        y=y[permutations]

        # Save the result to disk
        np.save('../saved_datasets/X_train_val.npy',X_train_val)
        np.save('../saved_datasets/X_train.npy',X_train)
        np.save('../saved_datasets/X_val.npy',X_val)
        np.save('../saved_datasets/X_test.npy',X_test)
        np.save('../saved_datasets/X.npy',X)
        np.save('../saved_datasets/y_train_val.npy',y_train_val)
        np.save('../saved_datasets/y_train.npy',y_train)
        np.save('../saved_datasets/y_val.npy',y_val)
        np.save('../saved_datasets/y_test.npy',y_test)
        np.save('../saved_datasets/y.npy',y)
    
    # Load data from disk
    else:
        X_train_val = np.load('../saved_datasets/X_train_val.npy')
        X_train=np.load('../saved_datasets/X_train.npy')
        X_val=np.load('../saved_datasets/X_val.npy')
        X_test=np.load('../saved_datasets/X_test.npy')
        X=np.load('../saved_datasets/X.npy')
        y_train_val=np.load('../saved_datasets/y_train_val.npy')
        y_train=np.load('../saved_datasets/y_train.npy')
        y_val=np.load('../saved_datasets/y_val.npy')
        y_test=np.load('../saved_datasets/y_test.npy')
        y=np.load('../saved_datasets/y.npy')

    print(f'X_train_val,y_train_val shapes: {X_train_val.shape}, {y_train_val.shape}')
    print(f'X_train,y_train shapes: {X_train.shape}, {y_train.shape}')
    print(f'X_val,y_val shapes: {X_val.shape}, {y_val.shape}')
    print(f'X_test,y_test shapes: {X_test.shape}, {y_test.shape}')
    print(f'X,y shapes: {X.shape}, {y.shape}')
    return X_train_val, X_train,X_val,X_test,X,y_train_val,y_train,y_val,y_test,y