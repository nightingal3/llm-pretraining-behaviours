####################################################################################################

# Copyright (c) 2021-, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####################################################################################################

#
# Note: If when running this conversion script you're getting an exception:
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# you need to tell python where to find the clone of Megatron-LM, e.g.:
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# if you already have it cloned elsewhere, simply adjust the path to the existing path
#
# If the training was done using a Megatron-LM fork, e.g.,
# https://github.com/microsoft/Megatron-DeepSpeed/ then chances are that you need to have that one
# in your path, i.e., /path/to/Megatron-DeepSpeed/
#

import argparse
import os
import re
import zipfile
import yaml

import torch

from transformers import AutoTokenizer, LlamaConfig


####################################################################################################


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


####################################################################################################


def convert_megatron_checkpoint(args, input_state_dict, config):
    # The converted output model.
    output_state_dict = {}

    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // heads

    kv_heads = config.num_key_value_heads

    # Megatron-LM checkpoint version
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # The model.
    model = input_state_dict["model"]
    # The language model.
    lm = model["language_model"]
    # The embeddings.
    embeddings = lm["embedding"]

    # The word embeddings.
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # Truncate the embedding table to vocab_size rows.
    # word_embeddings = word_embeddings[: config.vocab_size, :]
    output_state_dict["model.embed_tokens.weight"] = word_embeddings.clone()

    # The transformer.
    transformer = lm["encoder"]

    # The regex to extract layer names.
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # The simple map of names for "automated" rules.
    megatron_to_transformers = {
        "self_attention.dense": ".self_attn.o_proj.",
        "mlp.dense_4h_to_h": ".mlp.down_proj.",
        "input_layernorm": ".input_layernorm.",
        "post_attention_layernorm": ".post_attention_layernorm."
    }

    # Extract the layers.
    for key, val in transformer.items():
        # Match the name.
        m = layer_re.match(key)

        # Stop if that's not a layer
        if m is None:
            break

        # The index of the layer.
        layer_idx = int(m.group(1))
        # The name of the operation.
        op_name = m.group(2)
        # Is it a weight or a bias?
        weight_or_bias = m.group(3)

        # The name of the layer.
        layer_name = f"model.layers.{layer_idx}"

        # Transpose the QKV matrix.
        if op_name == "self_attention.query" and weight_or_bias == "weight":
            out_val = fix_query_key_value_ordering(
                val,
                checkpoint_version,
                1,
                heads,
                hidden_size_per_head,
            )
            output_state_dict[layer_name + ".self_attn.q_proj.weight"] = out_val

        elif op_name == "self_attention.key_value" and weight_or_bias == "weight":
            out_val = fix_query_key_value_ordering(
                val,
                checkpoint_version,
                2,
                kv_heads,
                hidden_size_per_head,
            )
            k_val, q_val = torch.split(out_val, out_val.shape[0] // 2, 0)
            output_state_dict[layer_name + ".self_attn.k_proj.weight"] = k_val.clone()
            output_state_dict[layer_name + ".self_attn.v_proj.weight"] = q_val.clone()

        # Transpose the weights.
        elif op_name == "mlp.dense_h_to_4h":
            up_val, gate_val = val.chunk(2, 0)
            up_val, gate_val = up_val.contiguous(), gate_val.contiguous()
            output_state_dict[layer_name + ".mlp.up_proj.weight"] = up_val.clone()
            output_state_dict[layer_name + ".mlp.gate_proj.weight"] = gate_val.clone()
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.clone()
            #if len(val.shape) == 2:
            #    output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # Copy the bias.
        elif weight_or_bias == "bias":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val.clone()

    # The final layernorm.
    output_state_dict["model.norm.weight"] = transformer["final_layernorm.weight"].clone()

    # Output vocabulary
    output_state_dict["lm_head.weight"] = lm["output_layer"]["weight"].clone()

    # It should be done!
    return output_state_dict


####################################################################################################


def main():
    # Create the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "--config-yml",
        type=str,
        help="A config yml file describing the pre-trained model.",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        type=str
    )
    parser.add_argument(
        "--verify-checkpoint",
        action="store_true",
        help="Verify the checkpoint matches the config (creates a new model and compares the state_dict).",
    )
    args = parser.parse_args()

    # Extract the basename.
    basename = os.path.dirname(args.path_to_checkpoint) \
               if args.output_folder is None \
               else args.output_folder

    # Load the model.
    # the .zip is very optional, let's keep it for backward compatibility
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    # load yaml config
    with open(args.config_yml, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)

    config = LlamaConfig(
            vocab_size=32000,
            max_position_embeddings=yml_config["seq_length"],
            hidden_size=yml_config["hidden_size"],
            num_hidden_layers=yml_config["num_layers"],
            num_attention_heads=yml_config["num_attention_heads"],
            num_key_value_heads=yml_config["num_kv_heads"],
            intermediate_size=yml_config["ffn_hidden_size"],
    )
    config.architectures = ["LlamaForCausalLM"]

    # Convert.
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    if args.verify_checkpoint:
        print("Verifying")
        from transformers import LlamaForCausalLM
        test_model = LlamaForCausalLM(config)
        test_model.load_state_dict(output_state_dict)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    
    tokenizer_name = "NousResearch/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_class = type(tokenizer).__name__
    config.tokenizer_class = tokenizer_class

    # Store the config to file.
    print("Saving config")
    config.save_pretrained(basename)

    # Save tokenizer based on args
    print(f"Adding {tokenizer_class} tokenizer files")
    tokenizer.save_pretrained(basename)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


####################################################################################################

if __name__ == "__main__":
    main()

####################################################################################################
