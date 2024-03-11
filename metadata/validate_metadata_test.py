import json
import os
from jsonschema import validate
from jsonschema.exceptions import ValidationError
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_METADATA_DIR = os.path.join(BASE_DIR, "model_metadata")
SCHEMA_FILE_PATH = os.path.join(BASE_DIR, "model_metadata_schema.json")

# Load the schema file once and use it for all validations
with open(SCHEMA_FILE_PATH, "r") as schema_file:
    schema = json.load(schema_file)


def get_json_files(directory):
    """Retrieve a list of JSON files from the specified directory."""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".json")
    ]


@pytest.mark.parametrize("json_file", get_json_files(MODELS_METADATA_DIR))
def test_validate_json_against_schema(json_file):
    """Test each JSON file in the directory against the schema."""
    with open(json_file, "r") as file:
        json_data = json.load(file)

    # Validate the JSON data against the schema
    try:
        validate(instance=json_data, schema=schema)
    except ValidationError as e:
        pytest.fail(f"Validation failed for '{json_file}': {e.message}")
