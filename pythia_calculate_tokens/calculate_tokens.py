import numpy as np
import json
import argparse

def get_text(batch_id):
    '''
    Get the text for a given batch_id
    '''
    with open("/home/lmarinov/pythia/utils/20B_tokenizer.json", "r") as f: 
        j = json.load(f)
    d = {v: k for k, v in j['model']['vocab'].items()}
    return "".join([d[i] for i in indexes[batch_id]]) #.replace('Ġ', ' ')

def get_tokens(batch_id):
    '''
    Get the tokens for a given batch_id
    Output is of type: numpy.ndarray where dtype=uint16
    '''
    return indexes[batch_id]

def get_all_tokens(batch_id):
    '''
    Get all tokens seen up to a given batch_id (exclusive)
    Output is of type: numpy.ndarray where dtype=uint16
    '''
    return indexes[:batch_id].flatten()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="",
    )
    parser.add_argument(
        "--checkpoint_id",
        type=int,
        default=0
    )
    args = parser.parse_known_args()[0]
    # Each indexes[i] is an array of length 2049 of tokens, which were used for training
    # The token mappings come from https://github.com/EleutherAI/pythia/blob/main/utils/20B_tokenizer.json
    indexes = np.load("/data/datasets/huggingface/eleutherai/pile-deduped-pythia-preshuffled/indices64.npy")
    if (args.checkpoint_id < 0 or args.checkpoint_id > 63):
        raise Exception("checkpoint_id out of bounds")
    all_toks = get_all_tokens(args.checkpoint_id * 1024)
    # print(all_toks)
    # print(len(all_toks))
    np.save('tokens', all_toks)
