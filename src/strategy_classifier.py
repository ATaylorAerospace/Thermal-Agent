"""
XGBoost classifier for Passive/Active/Hybrid strategy prediction.

Author: A Taylor
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


class StrategyClassifier:
    """XGBoost-based classifier for thermal mitigation strategy prediction.

    Predicts Passive, Active, or Hybrid strategies based on material,
    instrument, environment, and thermal effect features.

    Author: A Taylor
    """

    FEATURE_COLS = ["material_name", "instrument", "environment_location", "thermal_effect"]

    def __init__(self):
        """Initialize the classifier and label encoders."""
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self.encoders = {col: LabelEncoder() for col in self.FEATURE_COLS}
        self.target_encoder = LabelEncoder()
        self._is_fitted = False

    def _check_is_fitted(self):
        """Verify the classifier has been trained or loaded before prediction.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Classifier is not fitted. Call train() or load() before predict()."
            )

    def prepare_features(self, df):
        """Label-encode categorical columns for model input.

        Args:
            df: DataFrame with raw categorical features.

        Returns:
            DataFrame with encoded features.
        """
        encoded = pd.DataFrame()
        for col in self.FEATURE_COLS:
            if col in df.columns:
                encoded[col] = self.encoders[col].fit_transform(df[col])
            else:
                logger.warning("Column %s not found in DataFrame", col)
                encoded[col] = 0
        return encoded

    def _encode_features(self, df):
        """Transform features using already-fitted encoders."""
        encoded = pd.DataFrame()
        for col in self.FEATURE_COLS:
            if col in df.columns:
                encoded[col] = self.encoders[col].transform(df[col])
            else:
                encoded[col] = 0
        return encoded

    def train(self, df):
        """Fit the classifier on the provided DataFrame.

        Args:
            df: DataFrame with feature columns and 'strategy_type' target.
        """
        logger.info("Preparing features for %d samples", len(df))
        X = self.prepare_features(df)
        y = self.target_encoder.fit_transform(df["strategy_type"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Training XGBoost classifier...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        y_pred = self.model.predict(X_test)
        target_names = self.target_encoder.classes_

        report = classification_report(y_test, y_pred, target_names=target_names)
        logger.info("Classification Report:\n%s", report)
        print(report)

        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion Matrix:\n%s", cm)

    def predict(self, material, instrument, environment, thermal_effect):
        """Predict strategy for a single input combination.

        Args:
            material: Chip material name.
            instrument: Instrument name.
            environment: Environment location.
            thermal_effect: Thermal effect type.

        Returns:
            Predicted strategy string.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        row = pd.DataFrame([{
            "material_name": material,
            "instrument": instrument,
            "environment_location": environment,
            "thermal_effect": thermal_effect,
        }])
        X = self._encode_features(row)
        pred = self.model.predict(X)
        return self.target_encoder.inverse_transform(pred)[0]

    def predict_proba(self, material, instrument, environment, thermal_effect):
        """Predict strategy probabilities for a single input combination.

        Args:
            material: Chip material name.
            instrument: Instrument name.
            environment: Environment location.
            thermal_effect: Thermal effect type.

        Returns:
            Dict mapping strategy name to probability.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()
        row = pd.DataFrame([{
            "material_name": material,
            "instrument": instrument,
            "environment_location": environment,
            "thermal_effect": thermal_effect,
        }])
        X = self._encode_features(row)
        proba = self.model.predict_proba(X)[0]
        classes = self.target_encoder.classes_
        return {cls: float(p) for cls, p in zip(classes, proba)}

    def save(self, path):
        """Save the trained model and encoders to disk.

        Args:
            path: File path for the saved model.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "encoders": self.encoders,
            "target_encoder": self.target_encoder,
        }, path)
        logger.info("Model saved to %s", path)

    def load(self, path):
        """Load a trained model and encoders from disk.

        Args:
            path: File path of the saved model.
        """
        data = joblib.load(path)
        self.model = data["model"]
        self.encoders = data["encoders"]
        self.target_encoder = data["target_encoder"]
        self._is_fitted = True
        logger.info("Model loaded from %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strategy classifier training. Author: A Taylor")
    parser.add_argument("--dataset", type=str, default="Taylor658/deep-space-optical-chip-thermal-dataset")
    parser.add_argument("--save_path", type=str, default="results/strategy_classifier.pkl")
    args = parser.parse_args()

    logger.info("Loading dataset: %s", args.dataset)
    ds = load_dataset(args.dataset, split="train")
    df = ds.to_pandas()

    classifier = StrategyClassifier()
    classifier.train(df)
    classifier.save(args.save_path)
    logger.info("Done — model saved to %s", args.save_path)
