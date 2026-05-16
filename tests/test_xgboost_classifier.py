"""Regression tests for XGBoost classifier support."""

import unittest

from omegaconf import OmegaConf
from rich.console import Console
from rich.theme import Theme
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from modules.classifier import SKClassifier


class XGBoostClassifierTests(unittest.TestCase):
    """Validate XGBoost is a supported first-class classifier option."""

    CONSOLE = Console(theme=Theme({"success": "green", "info": "blue", "path": "cyan"}))

    @staticmethod
    def _make_config(use_scaler: bool = True):
        """Build the minimal classifier config required by SKClassifier."""
        return OmegaConf.create({"model": {"use_scaler": use_scaler}})

    def test_xgb_pipeline_uses_xgboost_estimator(self) -> None:
        """The xgb classifier type should instantiate an XGBClassifier."""
        classifier = SKClassifier("xgb", self._make_config(), console=self.CONSOLE)

        self.assertEqual(classifier.name, "XGBoost (scaled)")
        self.assertIsInstance(
            classifier.pipeline.named_steps["classifier"], XGBClassifier
        )

    def test_xgb_fit_and_predict_proba(self) -> None:
        """XGBoost should fit and emit binary class probabilities."""
        features, labels = make_classification(
            n_samples=40,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=7,
        )

        classifier = SKClassifier(
            "xgb", self._make_config(use_scaler=False), console=self.CONSOLE
        )
        classifier.set_params(
            n_estimators=8,
            max_depth=2,
            learning_rate=0.3,
            random_state=7,
            n_jobs=1,
            verbosity=0,
        )
        classifier.fit(features, labels)

        probabilities = classifier.predict_proba(features)

        self.assertEqual(probabilities.shape, (40, 2))
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())

    def test_model_config_includes_xgb_defaults(self) -> None:
        """The default model config should expose XGBoost selection and tuning."""
        config = OmegaConf.load("configs/model/model.yaml")

        self.assertIn("xgb", config.classifier)
        self.assertIn("xgb", config.param_grids)
        self.assertEqual(list(config.param_grids.xgb.n_estimators), [50, 100, 200])


if __name__ == "__main__":
    unittest.main()
