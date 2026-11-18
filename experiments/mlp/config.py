"""Configuration dataclasses for PathSnap Person 4 MLP experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from experiments.knn.config import ResourceBudget

VALID_ACTIVATIONS = {"identity", "logistic", "tanh", "relu"}
VALID_SOLVERS = {"lbfgs", "sgd", "adam"}
VALID_LEARNING_RATES = {"constant", "invscaling", "adaptive"}


@dataclass(frozen=True)
class MLPConfig:
    """Configurable hyperparameters and execution settings for scikit-learn MLP."""

    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    activation: str = "relu"
    solver: str = "adam"
    alpha: float = 1e-4
    learning_rate: str = "constant"
    learning_rate_init: float = 1e-3
    max_iter: int = 50
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 10
    batch_size: int | str = 1024
    inference_batch_size: int = 50_000
    n_splits: int = 5
    random_state: int = 42
    max_train_rows_per_fold: int | None = None
    use_float32: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hidden_layer_sizes, tuple)
            or len(self.hidden_layer_sizes) == 0
            or any(not isinstance(h, int) or h < 1 for h in self.hidden_layer_sizes)
        ):
            raise ValueError(
                "hidden_layer_sizes must be a non-empty tuple of positive integers"
            )
        if self.activation not in VALID_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(VALID_ACTIVATIONS)}"
            )
        if self.solver not in VALID_SOLVERS:
            raise ValueError(f"solver must be one of {sorted(VALID_SOLVERS)}")
        if self.alpha < 0.0:
            raise ValueError("alpha must be >= 0.0")
        if self.learning_rate not in VALID_LEARNING_RATES:
            raise ValueError(
                f"learning_rate must be one of {sorted(VALID_LEARNING_RATES)}"
            )
        if self.learning_rate_init <= 0.0:
            raise ValueError("learning_rate_init must be > 0.0")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.solver == "lbfgs" and self.early_stopping:
            raise ValueError(
                "early_stopping is only supported for stochastic solvers ('adam', 'sgd')"
            )
        if not (0.0 < self.validation_fraction < 1.0):
            raise ValueError("validation_fraction must be strictly between 0 and 1")
        if self.n_iter_no_change < 1:
            raise ValueError("n_iter_no_change must be >= 1")
        if self.batch_size != "auto" and (
            not isinstance(self.batch_size, int) or self.batch_size < 1
        ):
            raise ValueError("batch_size must be 'auto' or a positive integer")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be >= 1")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2 for GroupKFold")
        if (
            self.max_train_rows_per_fold is not None
            and self.max_train_rows_per_fold < 10
        ):
            raise ValueError("max_train_rows_per_fold must be >= 10")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["hidden_layer_sizes"] = list(self.hidden_layer_sizes)
        return payload


__all__ = ["MLPConfig", "ResourceBudget"]
