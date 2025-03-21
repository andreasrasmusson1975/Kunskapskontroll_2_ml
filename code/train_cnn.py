"""
MNIST RF Training and Evaluation Script.

This script automates the training, evaluation, and saving of a Support
Vector Machine classifier for the MNIST dataset. It follows these steps:

1. Fetch Data - Loads the MNIST dataset from disk using fetch().
2. Train Model - Trains the RF model on the training set (fit()).
3. Refit for validation Evaluation - Retrains the model before evaluation (refit_for_validation_set()).
4. Evaluate Model - Computes accuracy ,confidence interval and confusion_matrix (evaluate()).
5. Refit for test evaluation
6. Evaluate once more on the test set (refit for test_set)
7. Final Training - Re-trains the model on the full dataset (refit_for_final_model()).
8. Save Model - Saves the trained model to disk (save_model()).

Progress updates are displayed using tqdm.
"""

# Perform necessary imports
from mnist_cnn_classifier import MnistCnnClassifier
from data_fetcher import fetch
from tqdm import tqdm
import os
import time

# Load the dataset
X_train_val, X_train,X_val,X_test,X,y_train_val,y_train,y_val,y_test,y = fetch(from_disk=True)

# Initialize the classifier
cnn = MnistCnnClassifier()
# Perform the initial fit
#tqdm.write('\nInitial fit...\n')
#cnn.fit(X_train,y_train)

# Re-train for evaluation on the validation set
tqdm.write('Refitting for evaluation on validation set...\n')
cnn.refit_for_validation_set(X_train,y_train)

# Perform evaluation
tqdm.write('Performing evaluation on validation set...\n')
cnn.evaluate(X_val,y_val,validation=True)

#Re-train for evaluation on the test set
tqdm.write('Refitting for evaluation on test set...\n')
cnn.refit_for_test_set(X_train_val,y_train_val)

# Perform evaluation
tqdm.write('Performing evaluation on test set...\n')
cnn.evaluate(X_test,y_test,validation=False)
# Re-train on the full dataset
tqdm.write('Refitting for final model...\n')
cnn.refit_for_final_model(X,y)

# Save the model
tqdm.write('Saving model...\n')
cnn.save_model()

tqdm.write('Done.')