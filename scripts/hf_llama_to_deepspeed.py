import deepspeed
import torch
import yaml

from torch.optim import SGD

from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import save_checkpoint
from megatron.initialize import initialize_megatron
from megatron.training import get_model

from pretrain_gpt import model_provider

from transformers import LlamaForCausalLM

def main():
    initialize_megatron(extra_args_provider=add_load_hf_args,
                        args_defaults={})

    args = get_args()
    assert args.deepspeed, "This script expects deepspeed to be enabled."

    # Set up model and load checkpoint
    [ model ] = get_model(model_provider, wrap_with_ddp=False)

    # Use SGD as optimizer since we don't need to train,
    # and we don't want to add optimizer state.
    optimizer = SGD(model.parameters(), lr=args.lr)
    opt_param_scheduler = None
    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        args=args,
        lr_scheduler=opt_param_scheduler,
        mpu=mpu if args.no_pipeline_parallel else None
    )
    
    hf_model = LlamaForCausalLM.from_pretrained(args.hf_model, device_map="cpu")
    # Set model state.
    set_preprocess_state(args, model.module, hf_model)
    set_postprocess_state(args, model.module, hf_model)
    for layer_idx in range(args.num_layers):
        set_layer_state(args, model.module, hf_model, layer_idx)

    iteration = 1
    model = [model]
    save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def set_preprocess_state(args, model, hf_model):
    '''Set embedding params.'''
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = args.tensor_model_parallel_size

    full_weight = hf_model.model.embed_tokens.weight

    # Chunk to obtain the same behaviour as in 
    # Megatron-DeepSpeed/tools/checkpoint_saver_megatron.py
    shards = torch.chunk(full_weight, tp_size, dim=0)

    shard = shards[tp_rank]

    model.language_model.embedding.word_embeddings.weight.data.copy_(shard)


def set_postprocess_state(args, model, hf_model):
    '''Set output layer & norm params.'''
    model.language_model.encoder.final_layernorm.weight.data.copy_(
        hf_model.model.norm.weight,
    )

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = args.tensor_model_parallel_size

    full_lm_head_weight = hf_model.lm_head.weight

    # Chunk to obtain the same behaviour as in 
    # Megatron-DeepSpeed/tools/checkpoint_saver_megatron.py
    shards = torch.chunk(full_lm_head_weight, tp_size, dim=0)

    shard = shards[tp_rank]

    model.language_model.output_layer.weight.data.copy_(shard)

def set_layer_state(args, model, hf_model, layer_idx):
    '''Set transformer layer params.'''

    layer = model.language_model.encoder.layers[layer_idx]
    hf_layer = hf_model.model.layers[layer_idx]

    set_attn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer, layer_idx)
    layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight)
    layer.post_attention_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight)

def set_attn_state(args, layer, hf_layer):
    '''Set self-attention params.'''

    # Get attention layer & state.
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp = args.tensor_model_parallel_size
    nh = args.num_attention_heads
    ng = args.num_key_value_heads
    dim = args.kv_channels
    hidden_size = args.hidden_size
    assert nh % ng == 0

    # Reshape loaded weights.
    qkv = torch.cat([ 
        hf_attn.q_proj.weight.reshape((ng, dim*nh//ng, hidden_size)),
        hf_attn.k_proj.weight.reshape((ng, dim, hidden_size)),
        hf_attn.v_proj.weight.reshape((ng, dim, hidden_size)),
    ], dim=1).reshape((-1, args.hidden_size))

    qkv_shards = torch.chunk(qkv, tp, dim=0)
    qkv_shard = qkv_shards[tp_rank]

    # Copy weights (re-order dimensions for Megatron).
    attn.query_key_value.weight.data.copy_(qkv_shard)

    dense = hf_attn.o_proj.weight
    dense_shards = torch.chunk(dense, tp, dim=1)
    dense_shard = dense_shards[tp_rank]

    attn.dense.weight.data.copy_(dense_shard)


def set_mlp_state(args, layer, hf_layer, layer_idx):
    '''Set MLP params.'''

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp = args.tensor_model_parallel_size

    mlp = layer.mlp
    hf_mlp = hf_layer.mlp

    gate_proj = hf_mlp.gate_proj.weight.data
    up_proj = hf_mlp.up_proj.weight.data

    gate_proj_shards = torch.chunk(gate_proj, tp, dim=0)
    up_proj_shards = torch.chunk(up_proj, tp, dim=0)

    dense_h_to_4h_shards = [torch.cat(weights, dim=0) for weights in zip(gate_proj_shards, up_proj_shards)]
    dense_h_to_4h = dense_h_to_4h_shards[tp_rank]
    mlp.dense_h_to_4h.weight.data.copy_(dense_h_to_4h)

    down_proj = hf_mlp.down_proj.weight

    down_proj_shards = torch.chunk(down_proj, tp, dim=1)

    down_proj_shard = down_proj_shards[tp_rank]

    mlp.dense_4h_to_h.weight.data.copy_(down_proj_shard)


def add_load_hf_args(parser):
    group = parser.add_argument_group(title='hf-model')

    group.add_argument("--hf-model", type=str, required=True,
                       help='Path to the checkpoint to load.')
    return parser

if __name__ == "__main__":
    main()
