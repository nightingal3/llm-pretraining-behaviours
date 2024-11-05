import pytest
import pandas as pd
from pathlib import Path
from model_metadata_db import AnalysisStore


@pytest.fixture
def test_paths():
    base_dir = Path(__file__).parent
    return {
        "model_path": str(base_dir / "test_data/model.csv"),
        "dataset_path": str(base_dir / "test_data/dataset.csv"),
        "scores_path": str(base_dir / "test_data/scores.csv"),
    }


@pytest.fixture
def db():
    db = AnalysisStore(":memory:")
    yield db


def test_model_import(db, test_paths):
    db.import_model_features_from_csv(test_paths["model_path"])

    # Test OLMo-7B features
    result = db.con.execute(
        """
        SELECT dimension, num_heads, total_params
        FROM model_annotations
        WHERE id = 'allenai/OLMo-7B'
    """
    ).df()
    breakpoint()
    assert len(result) == 1
    assert result.iloc[0]["dimension"] == 4096
    assert result.iloc[0]["num_heads"] == 32
    assert result.iloc[0]["total_params"] == 6888095744


def test_dataset_import(db, test_paths):
    db.import_dataset_features_from_csv(test_paths["dataset_path"])

    # Test OLMo-7B dataset features
    result = db.con.execute(
        """
        SELECT pretraining_summary_total_tokens_billions,
               pretraining_summary_percentage_web,
               pretraining_summary_percentage_code
        FROM dataset_info
        WHERE id = 'allenai/OLMo-7B'
    """
    ).df()

    assert len(result) == 1
    assert result.iloc[0]["pretraining_summary_total_tokens_billions"] == 2460.0
    assert result.iloc[0]["pretraining_summary_percentage_web"] == 82.0
    assert result.iloc[0]["pretraining_summary_percentage_code"] == 13.43


def test_scores_import(db, test_paths):
    db.import_scores_from_csv(test_paths["scores_path"])

    # Test Pythia-410m scores
    result = db.con.execute(
        """
        SELECT benchmark, setting, accuracy, accuracy_stderr
        FROM evaluation_results
        WHERE model_id = 'EleutherAI/pythia-410m'
        AND benchmark = 'arc:challenge'
    """
    ).df()

    assert len(result) == 1
    assert result.iloc[0]["setting"] == "25-shot"
    assert pytest.approx(result.iloc[0]["accuracy"]) == 0.23122866894197952
    assert pytest.approx(result.iloc[0]["accuracy_stderr"]) == 0.012320858834772283


def test_joined_query(db, test_paths):
    # Import all data
    db.import_model_features_from_csv(test_paths["model_path"])
    db.import_dataset_features_from_csv(test_paths["dataset_path"])
    db.import_scores_from_csv(test_paths["scores_path"])

    # Test joined query
    result = db.con.execute(
        """
        SELECT 
            m.id,
            m.dimension,
            m.total_params,
            d.pretraining_summary_total_tokens_billions
        FROM model_annotations m
        JOIN dataset_info d ON m.id = d.id
        ORDER BY m.total_params DESC
    """
    ).df()

    assert len(result) == 3
    assert result.iloc[0]["id"] == "allenai/OLMo-7B"
    assert result.iloc[0]["dimension"] == 4096
    assert result.iloc[0]["total_params"] == 6888095744
    assert result.iloc[0]["pretraining_summary_total_tokens_billions"] == 2460.0


def test_complete_import(db, test_paths):
    # Import all data
    db.import_model_features_from_csv(test_paths["model_path"])
    db.import_dataset_features_from_csv(test_paths["dataset_path"])
    db.import_scores_from_csv(test_paths["scores_path"])

    # Verify counts
    model_count = db.con.execute("SELECT COUNT(*) FROM model_annotations").fetchone()[0]
    dataset_count = db.con.execute("SELECT COUNT(*) FROM dataset_info").fetchone()[0]

    assert model_count == 3
    assert dataset_count == 3

    # Verify each model has expected data
    models = ["allenai/OLMo-7B", "EleutherAI/pythia-410m", "cerebras/Cerebras-GPT-2.7B"]
    for model_id in models:
        result = db.con.execute(
            """
            SELECT 
                m.id,
                m.dimension IS NOT NULL as has_model_features,
                d.pretraining_summary_total_tokens_billions IS NOT NULL as has_dataset_features
            FROM model_annotations m
            LEFT JOIN dataset_info d ON m.id = d.id
            WHERE m.id = ?
        """,
            [model_id],
        ).df()

        assert len(result) == 1
        assert result.iloc[0]["has_model_features"]
        assert result.iloc[0]["has_dataset_features"]
