import json
import os
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARCHITECTURE_METADATA_DIR = os.path.join(BASE_DIR, "architecture_metadata")
ARCHITECTURE_SCHEMA_FILE_PATH = os.path.join(
    BASE_DIR, "architecture_metadata_schema.json"
)

RESULTS_METADATA_DIR = os.path.join(BASE_DIR, "model_scores")
RESULTS_SCHEMA_FILE_PATH = os.path.join(BASE_DIR, "model_results_schema.json")


DATASET_METADATA_DIR = os.path.join(BASE_DIR, "dataset_metadata")
DATASET_SCHEMA_FILE_PATH = os.path.join(BASE_DIR, "dataset_metadata_schema.json")

# Load the schema file once and use it for all validations
with open(ARCHITECTURE_SCHEMA_FILE_PATH, "r") as schema_file:
    ARCHITECTURE_SCHEMA = json.load(schema_file)
with open(RESULTS_SCHEMA_FILE_PATH, "r") as schema_file:
    RESULTS_SCHEMA = json.load(schema_file)
with open(DATASET_SCHEMA_FILE_PATH, "r") as schema_file:
    DATASET_SCHEMA = json.load(schema_file)


def get_schema(json_file_name):
    if ARCHITECTURE_METADATA_DIR in json_file_name:
        return ARCHITECTURE_SCHEMA
    elif DATASET_METADATA_DIR in json_file_name:
        return DATASET_SCHEMA
    else:
        return RESULTS_SCHEMA


def get_json_files(directory):
    """Retrieve a list of JSON files from the specified directory."""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".json")
    ]


@pytest.mark.parametrize(
    "json_file",
    get_json_files(ARCHITECTURE_METADATA_DIR)
    + get_json_files(RESULTS_METADATA_DIR)
    + get_json_files(DATASET_METADATA_DIR),
)
def test_validate_json_against_schema(json_file):
    """Test each JSON file in the directory against the schema."""
    with open(json_file, "r") as file:
        json_data = json.load(file)

    # Validate the JSON data against the schema
    try:
        validate(instance=json_data, schema=get_schema(json_file))
    except ValidationError as e:
        pytest.fail(f"Validation failed for '{json_file}': {e.message}")
