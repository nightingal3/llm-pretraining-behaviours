"""
Convert Megatron-DeepSpeed checkpoints to huggingface weights.

Adapted from
https://github.com/epfLLM/Megatron-LLM/blob/main/weights_conversion/megatron_to_hf.py
"""
# Copyright 2022 EleutherAI and The HuggingFace Inc. team. All rights reserved.
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
import gc
import yaml
import re
import os
import sys
import json
import warnings
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from argparse import ArgumentParser, Namespace
sys.path.append(str(Path(__file__).parent.parent.absolute()))  # megatron is importable

import torch
from tqdm.auto import trange
from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer


def permute_qkv(qkv_w: torch.Tensor, dim: int, n_heads: int,
                n_heads_kv: int, revert: bool = False) -> torch.Tensor:

    def permute(x):
        if revert:
            return x.view(head_dim//2, 2, dim).transpose(0, 1).reshape(head_dim, dim)
        return x.view(2, head_dim//2, dim).transpose(0, 1).reshape(head_dim, dim)

    head_dim = dim//n_heads
    n_qs_per_kv = n_heads//n_heads_kv
    n_groups = qkv_w.size(0)//head_dim//(n_qs_per_kv + 2)
    groups = torch.chunk(qkv_w, n_groups, dim=0)
    new = []
    for group in groups:
        *qs, k, v = torch.split(group, head_dim, dim=0)
        assert len(qs) == n_qs_per_kv, f"{len(qs)}, {n_qs_per_kv}"
        new += list(map(permute, qs)) + [permute(k), v]
    return torch.cat(new, dim=0)


def update_checkpoint(input_dir: Path, output_dir: Path, overwrite_ok: bool = False):
    # make sure megatron is importable
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))


    # prepare output dir
    if output_dir.exists():
        if not overwrite_ok:
            raise FileExistsError(f"Output directory {output_dir} already exists")
        print(f"Removing {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    # determine realease
    with open(input_dir/"latest_checkpointed_iteration.txt") as f:
        it = f.read()
    print("Updating weights of iteration", it)
    with open(output_dir/"latest_checkpointed_iteration.txt", "w+") as f:
        f.write(it)
    if it != "release":
        it = f"iter_{int(it):07d}"
    (output_dir/it).mkdir()

    # convert weights
    for fname in tqdm(list((input_dir/it).iterdir())):
        checkpoint = torch.load(fname/"model_optim_rng.pt", map_location="cpu")
        args = checkpoint["args"]
        args = (args.hidden_size, args.num_attention_heads,
                args.num_attention_heads_kv)
        if "transformer" in checkpoint["model"]["language_model"]:
            key = "transformer"
            attn_key = "attention"
        else:
            key = "encoder"
            attn_key = "self_attention"
        states = checkpoint["model"]["language_model"][key]
        for name, weight in states.items():
            if re.match(rf"^layers\.[0-9]+\.{attn_key}\.query_key_value\.weight$", name):
                states[name] = permute_qkv(weight, *args)
        (output_dir/it/fname.stem).mkdir()
        torch.save(checkpoint, output_dir/it/fname.stem/"model_optim_rng.pt")


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def convert_wqkv(llama_mega, layer_idx=0, n_heads=32, n_heads_kv=8):
    qkv_w = llama_mega["transformer"][f'layers.{layer_idx}.attention.query_key_value.weight']
    n_hidden = qkv_w.size(1)
    hidden_dim = n_hidden//n_heads
    qkv_w = permute_qkv(qkv_w, n_hidden, n_heads, n_heads_kv, revert=True)

    n_qs_per_kv = n_heads//n_heads_kv
    n_groups = qkv_w.size(0)//hidden_dim//(n_qs_per_kv + 2)
    qkv_w = list(torch.split(qkv_w, hidden_dim, dim=0))

    wq, wk, wv = [], [], []
    for group in range(n_groups):
        for qs in range(n_qs_per_kv):
            wq.append(qkv_w[0])
            del qkv_w[0]
        wk.append(qkv_w[0])
        del qkv_w[0]
        wv.append(qkv_w[0])
        del qkv_w[0]
    assert len(qkv_w) == 0

    wq = torch.concat(wq, dim=0)
    wk = torch.concat(wk, dim=0)
    wv = torch.concat(wv, dim=0)
    return wq, wk, wv


def convert_ffn(llama_mega, layer_idx=0, n_dense=11008):
    mega_ffn = llama_mega["transformer"][f'layers.{layer_idx}.mlp.dense_h_to_4h.weight']
    ffn_w3, ffn_w1 = mega_ffn.split(n_dense, dim=0)
    return ffn_w1, ffn_w3


def write_llama_model(
        path_to_checkpoint,
        config_yml,
        tokenizer,
        output_dir,
        num_output_shards: int=2,
        norm_eps: float=1e-05,
        rope_theta: float=1e4
):

    # Preliminaries
    print(f"Fetching all parameters from the checkpoint at {path_to_checkpoint}.")
    input_state_dict = torch.load(path_to_checkpoint, map_location="cpu")
    # Load weights

    if "model" not in input_state_dict:
        # unflatten the state dict
        # TODO: kinda hard-coded atm 
        loaded = {"embedding": {}, "encoder": {}}
        for k, v in input_state_dict.items():
            if k == "language_model.embedding.word_embeddings.weight":
                loaded["embedding"]["word_embeddings"] = {"weight": v}
            if k == "language_model.output_layer.weight":
                loaded["lm_head"] = v
            elif k.startswith("language_model.encoder"):
                loaded["encoder"][k.replace("language_model.encoder.", "")] = v
        
    else:
        loaded = input_state_dict['model']['language_model']

    if 'transformer' not in loaded:  # normalize key names
        loaded["transformer"] = loaded.pop("encoder")
        for key in list(loaded["transformer"].keys()):
            loaded["transformer"][key.replace("self_attention", "attention")] = loaded["transformer"].pop(key)
        loaded["embedding"]["word_embeddings.weight"] = loaded["embedding"].pop("word_embeddings")["weight"]

    # Load arguments

    # load yaml config
    with open(config_yml, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)

    n_layers = yml_config["num_layers"]
    n_heads = yml_config["num_attention_heads"]
    n_heads_kv = yml_config["num_kv_heads"]
    n_dense = yml_config["ffn_hidden_size"]
    n_hidden = yml_config["hidden_size"]
    hidden_per_head = n_hidden // n_heads
    intermediate_size = yml_config["ffn_hidden_size"]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, hidden_per_head, 2).float() / hidden_per_head))

    print('Llama-Megatron Loaded!')
    param_count = 0
    index_dict = {"weight_map": {}}
        
    # Start conversion
    with TemporaryDirectory() as tmp_model_path:
        print(f'Weighted Converting for {n_layers} layers...')
        for layer_i in range(n_layers):
            filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
            wq_proj, wk_proj, wv_proj = convert_wqkv(llama_mega=loaded, 
                                          layer_idx=layer_i, n_heads=n_heads,
                                          n_heads_kv=n_heads_kv)
            ffn_w1, ffn_w3 = convert_ffn(llama_mega=loaded, 
                                        layer_idx=layer_i, 
                                        n_dense=n_dense)
            state_dict = {
                f"model.layers.{layer_i}.self_attn.q_proj.weight": wq_proj,
                f"model.layers.{layer_i}.self_attn.k_proj.weight": wk_proj,
                f"model.layers.{layer_i}.self_attn.v_proj.weight": wv_proj,
                f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded["transformer"][f"layers.{layer_i}.attention.dense.weight"],
                f"model.layers.{layer_i}.mlp.gate_proj.weight": ffn_w1,
                f"model.layers.{layer_i}.mlp.down_proj.weight": loaded["transformer"][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"],
                f"model.layers.{layer_i}.mlp.up_proj.weight": ffn_w3,
                f"model.layers.{layer_i}.input_layernorm.weight": loaded["transformer"][f"layers.{layer_i}.input_layernorm.weight"],
                f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded["transformer"][f"layers.{layer_i}.post_attention_layernorm.weight"],
                f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq": inv_freq
            }

            for k, v in state_dict.items():
                index_dict["weight_map"][k] = filename
                param_count += v.numel()
            torch.save(state_dict, os.path.join(tmp_model_path, filename))
            print(f'Sharded file saved to {filename}')

        filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
        state_dict = {
            "model.norm.weight": loaded["transformer"]['final_layernorm.weight'],
            "lm_head.weight": loaded['lm_head'],
            "model.embed_tokens.weight": loaded['embedding']["word_embeddings.weight"]
        }

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch_dtype = state_dict["lm_head.weight"].dtype
        torch.save(state_dict, os.path.join(tmp_model_path, filename))
        print(f'Sharded file saved to {filename}')

        # Write configs and save
        index_dict["metadata"] = {"total_size": param_count * 2}
        write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))
        config = LlamaConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=n_hidden,
            intermediate_size=intermediate_size,
            num_attention_heads=n_heads,
            num_hidden_layers=n_layers,
            rms_norm_eps=norm_eps,
            num_key_value_heads=n_heads_kv,
            max_position_embeddings=yml_config["seq_length"],
        )
        config.save_pretrained(tmp_model_path)

        # Make space so we can load the model properly now.
        del state_dict
        del loaded
        gc.collect()

        print("Loading the checkpoint in a Llama model...")
        model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch_dtype)
        # Avoid saving this as part of the config.
        del model.config._name_or_path

    print("Saving in the Transformers format.")
    max_num_params_per_shard = param_count*2 // max(1,(num_output_shards-1))
    model.save_pretrained(output_dir, max_shard_size=max_num_params_per_shard)


def main():
    # Create the argument parser.
    parser = ArgumentParser()
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    parser.add_argument(
        "output_folder",
        type=str,
        help="The output folder where the huggingface weights should be stored.",
    )
    parser.add_argument(
        "--config-yml",
        type=str,
        help="A config yml file describing the pre-trained model.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        help="The name of the HF tokenizer to use.",
    )
    parser.add_argument(
        "--model",
        choices={"llama", "llama2", "codellama"},
        default="llama2",
        type=str,
    )

    
    args = parser.parse_args()

    # get_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if args.model in {"llama", "llama2", "codellama"}:
        eps = 1e-6 if args.model == "llama" else 1e-5
        rope_theta = 1e6 if args.model == "codellama" else 1e4
        write_llama_model(
            args.path_to_checkpoint,
            args.config_yml,
            tokenizer,
            args.output_folder,
            num_output_shards=1,
            norm_eps=eps,
            rope_theta=rope_theta,
        )

    # Save tokenizer based on args
    print(f"Adding {tokenizer.__class__.__name__} tokenizer files")
    tokenizer.save_pretrained(args.output_folder)

if __name__ == "__main__":
    main()