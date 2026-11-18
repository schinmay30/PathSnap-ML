"""Model and HMM/Viterbi artifact discovery interface.

This module defines the boundary between FastAPI services and trained artifacts
exported into `artifacts/models/` (KNN, MLP, Person 3's final tuned model, and
the HMM/Viterbi decoder configuration).

No fake or synthetic models are constructed when artifacts are absent.
"""

from pathlib import Path

MODEL_ARTIFACT_PATTERNS: dict[str, tuple[str, ...]] = {
    "knn": ("knn*.joblib", "knn*.pkl"),
    "mlp": ("mlp*.joblib", "mlp*.pkl", "mlp*.pt"),
    "final": ("final*.joblib", "final*.pkl", "best_model*.joblib", "best_model*.pkl"),
}

HMM_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "hmm*.json",
    "hmm*.yaml",
    "hmm*.yml",
    "hmm*.joblib",
    "viterbi*.json",
)


def find_model_artifact(model_key: str, search_dirs: list[Path]) -> Path | None:
    """Discover a model, preferring the file declared by its matching manifest."""

    patterns = MODEL_ARTIFACT_PATTERNS.get(model_key, ())

    if model_key == "final":
        for directory in search_dirs:
            if not directory.is_dir():
                continue

            for manifest_path in sorted(directory.glob("viterbi*.json")):
                try:
                    import json

                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )

                    model_name = manifest.get("model_file")

                    if not isinstance(model_name, str):
                        continue

                    # Only accept filenames, not paths outside the directory.
                    if Path(model_name).name != model_name:
                        continue

                    candidate = directory / model_name

                    if candidate.is_file():
                        return candidate

                except (OSError, ValueError, TypeError):
                    continue

    for directory in search_dirs:
        if not directory.is_dir():
            continue

        for pattern in patterns:
            matches = sorted(
                p for p in directory.glob(pattern) if p.is_file()
            )

            if matches:
                return matches[0]

    return None

def find_hmm_artifact(search_dirs: list[Path]) -> Path | None:
    """Return the first real HMM/Viterbi config/artifact file, if present."""
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for pattern in HMM_ARTIFACT_PATTERNS:
            matches = sorted(p for p in directory.glob(pattern) if p.is_file())
            if matches:
                return matches[0]
    return None
