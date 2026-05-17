"""Regression tests for Decision Tree classifier support."""

from pathlib import Path
import unittest

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.theme import Theme
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from modules.classifier import SKClassifier


class DecisionTreeClassifierTests(unittest.TestCase):
    """Validate Decision Tree is a supported first-class classifier option."""

    CONSOLE = Console(theme=Theme({"success": "green", "info": "blue", "path": "cyan"}))

    @staticmethod
    def _make_config(use_scaler: bool = True) -> DictConfig:
        """Build the minimal classifier config required by SKClassifier."""
        return OmegaConf.create({"model": {"use_scaler": use_scaler}})

    def test_dt_pipeline_uses_decision_tree_estimator(self) -> None:
        """The dt classifier type should instantiate a DecisionTreeClassifier."""
        classifier = SKClassifier("dt", self._make_config(), console=self.CONSOLE)

        self.assertEqual(classifier.name, "Decision Tree (scaled)")
        self.assertIsInstance(
            classifier.pipeline.named_steps["classifier"], DecisionTreeClassifier
        )

    def test_dt_fit_and_predict_proba(self) -> None:
        """Decision Tree should fit successfully and emit class probabilities."""
        X, y = make_classification(
            n_samples=80,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=42,
        )
        classifier = SKClassifier(
            "dt", self._make_config(use_scaler=False), console=self.CONSOLE
        )

        classifier.fit(X, y)

        y_pred = classifier.predict(X)
        y_prob = classifier.predict_proba(X)

        self.assertEqual(y_pred.shape, (X.shape[0],))
        self.assertEqual(y_prob.shape, (X.shape[0], 2))

    def test_dt_grid_search_with_final_eval_tracks_best_params(self) -> None:
        """Grid search should work for Decision Tree classifiers."""
        X, y = make_classification(
            n_samples=90,
            n_features=8,
            n_informative=5,
            n_redundant=0,
            random_state=7,
        )
        classifier = SKClassifier(
            "dt", self._make_config(use_scaler=False), console=self.CONSOLE
        )

        metrics = classifier.grid_search_with_final_eval(
            X,
            y,
            param_grid={"max_depth": [2, 4], "class_weight": ["balanced"]},
            grid_search_cv=3,
            final_eval_cv=3,
            verbose=False,
        )

        self.assertIsNotNone(metrics.best_params)
        assert metrics.best_params is not None
        self.assertIn("max_depth", metrics.best_params)

    def test_model_config_includes_decision_tree_defaults(self) -> None:
        """Decision Tree should be present in the default model configuration."""
        model_config = OmegaConf.load(Path("configs/model/model.yaml"))

        self.assertIn("dt", list(model_config.classifier))
        self.assertIn("dt", model_config.param_grids)


if __name__ == "__main__":
    unittest.main()
