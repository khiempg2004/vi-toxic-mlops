from vitoxic import baseline
from vitoxic.metrics import classification_metrics

from .conftest import synthetic_frame


def test_baseline_learns_synthetic():
    train, test = synthetic_frame(60, seed=1), synthetic_frame(20, seed=2)
    _, res = baseline.train_and_eval(
        train["text"], train["label"], {"test": (test["text"], test["label"])}, C=4.0, max_iter=500, min_df=1
    )
    assert res["sets"]["test"]["macro_f1"] > 0.9


def test_baseline_main_writes_artifacts(project):
    baseline.main(["baseline.max_iter=300"])
    assert (project / "models" / "baseline" / "model.joblib").exists()
    assert (project / "reports" / "baseline_metrics.json").exists()


def test_metrics_invalid_counts_as_wrong():
    m = classification_metrics(["CLEAN", "HATE"], ["CLEAN", None])
    assert m["invalid_rate"] == 0.5 and m["accuracy"] == 0.5
