import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class AutoencoderModel:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_model()
        self.reconstruction_error_threshold = None
    
    def _build_model(self) -> Model:
        """Build autoencoder architecture"""
        # Input layer
        input_layer = Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(16, activation='relu')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(16, activation='relu')(encoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(32, activation='relu')(decoded)
        decoded = BatchNormalization()(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Create and compile model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32) -> None:
        """Train the autoencoder on normal merchant behavior"""
        logger.info("Training autoencoder model...")
        
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        # Calculate reconstruction error threshold
        reconstructions = self.model.predict(X_train)
        mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
        self.reconstruction_error_threshold = np.percentile(mse, 95)
        
        logger.info(f"Model trained. Reconstruction error threshold: {self.reconstruction_error_threshold}")
    
    def predict_anomalies(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies based on reconstruction error"""
        if self.reconstruction_error_threshold is None:
            raise ValueError("Model not trained. Run train() first.")
        
        reconstructions = self.model.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        
        # Calculate anomaly scores (0-1 range)
        anomaly_scores = mse / (self.reconstruction_error_threshold * 2)
        anomaly_scores = np.clip(anomaly_scores, 0, 1)
        
        # Binary predictions (anomaly or not)
        predictions = (mse > self.reconstruction_error_threshold).astype(int)
        
        return predictions, anomaly_scores 