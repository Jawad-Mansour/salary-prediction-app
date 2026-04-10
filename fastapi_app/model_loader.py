"""
Model loader - loads once at startup, reused for all requests
"""

import joblib
import numpy as np
from pathlib import Path


class ModelLoader:
    """Singleton class to load and hold the model and transformer"""
    
    _instance = None
    _model = None
    _transformer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        """Load the model and transformer from disk"""
        # CHANGED: Use v4 models
        model_path = Path("models/decision_tree_v4.pkl")
        transformer_path = Path("models/transformer_v4.pkl")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self._model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(self._model).__name__}")
        
        if transformer_path.exists():
            self._transformer = joblib.load(transformer_path)
            print(f"✅ Transformer loaded")
        else:
            print(f"⚠️ Transformer not found at {transformer_path}")
            self._transformer = None
    
    def predict(self, X):
        """Make prediction and inverse transform if needed"""
        prediction_transformed = self._model.predict(X)[0]
        
        if self._transformer is not None:
            prediction_array = self._transformer.inverse_transform(
                np.array([[prediction_transformed]])
            )
            prediction = float(prediction_array[0])
            return prediction
        
        return float(prediction_transformed)
    
    @property
    def is_loaded(self):
        return self._model is not None
    
    @property
    def model_type(self):
        return type(self._model).__name__ if self._model else None


# Global instance
model_loader = ModelLoader()