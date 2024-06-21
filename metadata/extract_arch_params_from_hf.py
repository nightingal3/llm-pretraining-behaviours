from transformers import AutoModel, AutoConfig
import torch

# Replace 'your-model-name' with the actual model name you're interested in
model_name = "jb723/LLaMA2-en-ko-7B-model"

# Load the model configuration
config = AutoConfig.from_pretrained(model_name)

# Load the model
model = AutoModel.from_pretrained(model_name)

# Extract total number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Try to deduce some of the specifics
# Note: These are heuristics and may not accurately reflect all models.
positional_embedding = "Unknown"
attention_variant = "Unknown"
biases = "Unknown"

# Checking for positional embeddings type (this is heuristic and may not be accurate for all models)
if hasattr(config, "max_position_embeddings"):
    positional_embedding = (
        "Absolute"  # Assuming absolute if max_position_embeddings is present
    )
if hasattr(config, "position_embedding_type"):
    positional_embedding = config.position_embedding_type  # More specific if available

# Check for biases in one of the model's linear layers as a proxy
for name, param in model.named_parameters():
    if "bias" in name:
        biases = "Present"
        break

# Assuming standard attention unless specific configurations indicate otherwise
attention_variant = "Standard"  # This is a simplistic assumption

print(f"Model Name: {model_name}")
print(f"Total Parameters: {total_params}")
print(f"Positional Embedding: {positional_embedding}")
print(f"Attention Variant: {attention_variant}")
print(f"Biases: {biases}")
