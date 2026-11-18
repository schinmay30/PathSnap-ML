"""Fixed tree baselines for candidate classification."""

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    forest = ExtraTreesClassifier(
        n_estimators=40, max_depth=18, min_samples_leaf=5,
        n_jobs=2, random_state=random_state,
    )
    boosting = HistGradientBoostingClassifier(
        early_stopping=False, random_state=random_state,
    )
    return {
        "extra_trees": Pipeline([("classifier", forest)]),
        "hist_gradient_boosting": Pipeline([("classifier", boosting)]),
    }
