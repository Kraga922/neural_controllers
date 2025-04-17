import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

class PlattCalibration:
    def __init__(self):
        self.calibrator = LogisticRegression(C=1.0, solver='lbfgs')
        
    def fit(self, predictions, true_labels):
        """
        Fit the calibration model
        
        Args:
            predictions: Raw uncalibrated predictions from your model (e.g. SVM scores)
            true_labels: True binary labels (0 or 1)
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(true_labels, torch.Tensor):
            true_labels = true_labels.cpu().numpy()
            
        # Fit logistic regression model
        self.calibrator.fit(predictions, true_labels)
        
        return self
    
    def calibrate(self, predictions):
        """
        Transform raw model outputs into calibrated probabilities
        
        Args:
            predictions: Raw uncalibrated predictions from your model
            
        Returns:
            Calibrated probabilities
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            
        predictions_reshaped = np.array(predictions).reshape(-1, 1)
        return self.calibrator.predict_proba(predictions_reshaped)[:, 1]
    
