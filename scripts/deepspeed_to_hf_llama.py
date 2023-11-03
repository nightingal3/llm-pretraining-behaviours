import argparse
import json
import torch
import yaml

from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig

from typing import Callable, TypedDict


def main():
    args = parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(yaml.dump(config))
    
    output_dir = Path(args.output_dir)

    n_layers = config["num_layers"]
    n_heads = config["num_attention_heads"]
    n_heads_kv = config["num_kv_heads"]
    n_hidden = config["hidden_size"]
    intermediate_size = config["ffn_hidden_size"]
    seq_length = config["seq_length"]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    llama_config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=tokenizer.vocab_size,
        hidden_size=n_hidden,
        intermediate_size=intermediate_size,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        rms_norm_eps=1e-5,
        num_key_value_heads=n_heads_kv,
        max_position_embeddings=seq_length,
    )

    ds_state_dict = torch.load(args.fp32_ckpt_path, map_location="cpu")

    hf_state_dict = {}

    io = init_io(hf_state_dict, ds_state_dict)
    
    write_hf_state_dict(config, io)
    
    filename = "pytorch_model.bin"
    states_save_file = output_dir / filename
    index_save_file = output_dir / "pytorch_model.bin.index.json"        
    
    index_dict = build_index_dict(hf_state_dict, filename)

    output_dir.mkdir(parents=True, exist_ok=True)

    llama_config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(index_save_file, "w") as f:
        json.dump(index_dict, f, indent=2)

    torch.save(hf_state_dict, states_save_file)

    del ds_state_dict
    del hf_state_dict
    del tokenizer

    # Test optimizer model loading
    _ = AutoTokenizer.from_pretrained(output_dir)
    _ = AutoModelForCausalLM.from_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-ckpt-path", type=str, required=True,
                        help="Path to checkpoint converted by deepspeed.")
    parser.add_argument("--config-path", type=str, required=True,
                        )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)

    return parser.parse_args()


GetFn = Callable[[str], torch.Tensor]
SetFn = Callable[[str, torch.Tensor], None]
CopyFn = Callable[[str, str], None]

@dataclass(frozen=True)
class StatesIO:
    get_fn: GetFn
    set_fn: SetFn

    @property
    def copy_fn(self):
        return lambda hf_key, ds_key: self.set_fn(hf_key, self.get_fn(ds_key))
    

class ModelConfig(TypedDict):
    hidden_size: int
    ffn_hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_query_groups: int
    seq_length: int
    num_kv_heads: int


def init_io(hf_state_dict, ds_state_dict) -> StatesIO:
    def get_fn(ds_key):
        assert ds_key in ds_state_dict, f"Key {ds_key} not found in DS state dict"
        return ds_state_dict[ds_key]

    def set_fn(hf_key, value):
        assert hf_key not in hf_state_dict, f"Key {hf_key} already in HF state dict"
        hf_state_dict[hf_key] = value

    return StatesIO(get_fn, set_fn)


def write_hf_state_dict(model_config: ModelConfig, io: StatesIO):
    num_layers = model_config["num_layers"]    

    hf_dict_update_embeddings(io)

    for i in range(num_layers):
        update_layer(i, model_config, io)

    update_final_layernorm(io)
    update_lm_head(io)


def hf_dict_update_embeddings(io: StatesIO):
    io.copy_fn(
        "model.embed_tokens.weight",
        "language_model.embedding.word_embeddings.weight",
    )

def update_layer(layer_idx: int, model_config: ModelConfig, io: StatesIO):
    ds_prefix = f"language_model.encoder.layers.{layer_idx}"
    hf_prefix = f"model.layers.{layer_idx}"

    update_attention(hf_prefix, ds_prefix, model_config, io)
    update_mlp(hf_prefix, ds_prefix, io)
    
    io.copy_fn(f"{hf_prefix}.input_layernorm.weight", f"{ds_prefix}.input_layernorm.weight")
    io.copy_fn(f"{hf_prefix}.post_attention_layernorm.weight", f"{ds_prefix}.post_attention_layernorm.weight")

def update_attention(
    hf_layer_prefix: str,
    ds_layer_prefix: str,
    model_config: ModelConfig,
    io: StatesIO,
):
    nh = model_config["num_attention_heads"]
    ng = model_config["num_kv_heads"]
    hidden_size = model_config["hidden_size"]
    
    hf_prefix = f"{hf_layer_prefix}.self_attn"
    ds_prefix = f"{ds_layer_prefix}.self_attention"

    qkv = io.get_fn(f"{ds_prefix}.query_key_value.weight")
    qkv = qkv.reshape((ng, -1, hidden_size))
    dim = qkv.shape[1] // (nh//ng + 2)

    q_proj = qkv[:, :dim*nh//ng, :]
    k_proj = qkv[:, dim*nh//ng:dim*nh//ng + dim, :]
    v_proj = qkv[:, dim*nh//ng + dim:, :]

    q_proj = q_proj.reshape((ng * dim*nh//ng, -1))
    k_proj = k_proj.reshape((ng * dim, -1))
    v_proj = v_proj.reshape((ng * dim, -1))

    io.set_fn(f"{hf_prefix}.q_proj.weight", q_proj)
    io.set_fn(f"{hf_prefix}.k_proj.weight", k_proj)
    io.set_fn(f"{hf_prefix}.v_proj.weight", v_proj)

    io.copy_fn(f"{hf_prefix}.o_proj.weight", f"{ds_prefix}.dense.weight")


def update_mlp(hf_prefix: str, ds_prefix: str, io: StatesIO):
    concatenated = io.get_fn(f"{ds_prefix}.mlp.dense_h_to_4h.weight")

    gate_proj = concatenated[:concatenated.shape[0]//2]
    up_proj = concatenated[concatenated.shape[0]//2:]

    io.set_fn(f"{hf_prefix}.mlp.gate_proj.weight", gate_proj)
    io.set_fn(f"{hf_prefix}.mlp.up_proj.weight", up_proj)
    
    io.copy_fn(f"{hf_prefix}.mlp.down_proj.weight", f"{ds_prefix}.mlp.dense_4h_to_h.weight")

def update_final_layernorm(io: StatesIO):
    io.copy_fn(
        "model.norm.weight",
        "language_model.encoder.final_layernorm.weight",
    )

def update_lm_head(io: StatesIO):
    io.copy_fn(
        "lm_head.weight",
        "language_model.output_layer.weight",
    )

def build_index_dict(hf_state_dict, filename: str):
    index_dict = {"weight_map": {}}

    param_count = 0
    for k, v in hf_state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    
    dtype = hf_state_dict["lm_head.weight"].dtype

    index_dict["metadata"] = {"total_size": param_count * dtype.itemsize}

    return index_dict


if __name__ == "__main__":
    main()