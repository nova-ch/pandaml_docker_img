
import os
import mlflow

import numpy as np
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Set the tracking URI to a local directory (Ensure mlruns directory is mounted properly via Docker)
mlflow.set_tracking_uri("file:///app/mlruns")

# Create a new experiment or use an existing one
experiment_name = "Demo_Experiment"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)

# Set the active experiment
mlflow.set_experiment(experiment_name)

# Enable auto-logging for MLflow
mlflow.tensorflow.autolog()


# Enable auto-logging for MLflow
mlflow.tensorflow.autolog()

# Generate some example data
X = np.random.rand(1000, 20)
y = np.random.rand(1000, 1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple sequential model
model = Sequential([
    Dense(64, activation='relu', input_shape=(20,)),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Start an MLflow run
with mlflow.start_run() as run:
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model
    loss, mae = model.evaluate(X_test, y_test, verbose=2)

    # Log metrics manually (in addition to autologging)
    mlflow.log_metric("loss", loss)
    mlflow.log_metric("mae", mae)

    # Save a custom artifact (for example, a plot or text file)
    artifact_file = "example_artifact.txt"
    with open(artifact_file, "w") as f:
        f.write("This is an example artifact")
    mlflow.log_artifact(artifact_file)

    # Output to check if everything went fine
    print(f"Model training and logging finished with run_id: {run.info.run_id}")

