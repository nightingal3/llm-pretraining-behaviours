import numpy as np
import stanza
from typing import List, Dict, Tuple, Any, Union, Optional, NewType

### Parse-based features (word + sentence level)
# TODO: map back to token-level
# * features that are easily calculated using stanza
## Constituency parse
# 1. Depth of word
# 2. Depth of tree
# *3. POS (UPOS, XPOS) TODO: map POS labels -> int
# *4. Number of words per sentence
# *5. Number of words in input 
# *6. Number of sentences in input (num_sentences)
## Dependency parse
# 1. Distance of word from immediate head
# 2. Distance of word to root

FeatureDict = NewType("FeatureDict", Dict[str, List[Union[str, int, float]]])


def get_const_parse_features(
    input_text: str,
    pipeline: stanza.Pipeline,
) -> FeatureDict:
    """
    Takes as input a chunk of text (paragraph, document) and stanza pipeline
    Lang for the pipeline should be 'en', processors are 'tokenize,pos,constituency'
    Returns a feature dict, where values are lists features per word
    """
    feature_dict = FeatureDict({
            "const_word_depth": [],
            "const_tree_depth": [],
            "upos_label": [],
            "xpos_label": [],
            "num_words_sentence": [],
            "num_words_input": [],
            "num_sentences_input": [],
        })

    def get_depth_features(
        tree: stanza.constituency,
    ) -> Tuple[List[int], List[int]]:
        def traverse_get_depth(
            tree,
            word_depths: List[int],
            depth: int,
        ):
            if len(tree.children) == 0:
                word_depths.append(depth)
            subtrees = tree.children
            depth += 1
            for subtree in subtrees:
                traverse_get_depth(subtree, word_depths, depth)

        word_depths = []
        tree = tree.children[0]
        traverse_get_depth(tree, word_depths, -1)
        max_depth = [max(word_depths)] * len(word_depths)
        return word_depths, max_depth

    # pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    processed_text = pipeline(input_text)
    num_words_input = 0
    for sentence in processed_text.sentences:
        num_words_input += len(sentence.words)
        for word in sentence.words:
            feature_dict.num_sentences_input.append(len(processed_text.sentences))
            feature_dict.num_words_sentence.append(len(sentence.words))
            feature_dict.upos_label.append(word.upos)
            feature_dict.xpos_label.append(word.xpos)
        word_depths, max_depth = get_depth_features(sentence.constituency)
        feature_dict.const_word_depth += word_depths
        feature_dict.const_tree_depth += max_depth
    feature_dict.num_words_input = [num_words_input] * num_words_input
    return feature_dict


def get_dep_parse_features(
    input_text: str,
    pipeline: stanza.Pipeline,
) -> FeatureDict:
    """
    Takes as input a chunk of text (paragraph, document) and stanza pipeline
    Lang for the pipeline should be 'en', processors are 'tokenize,mwt,pos,lemma,depparse'
    Returns a feature dict, where values are lists features per word
    """
    feature_dict = FeatureDict({
        "dist_to_head": [],
        "dist_to_root": []
    })
    processed_text = pipeline(input_text)
    for sentence in processed_text.sentences:
        root_dist = np.zeros(len(sentence.words), 1)
        head_dist = []
        root = -1
        for i, word in enumerate(sentence.words):
            root_dist[i] = word.id
            if word.head == 0:
                head_dist.append(0)
            else:
                head_dist.append(abs(word.id - word.head))
            if word.head == 0:
                root = word.id
        root_dist -= root
        feature_dict.dist_to_head += head_dist
        feature_dict.dist_to_root += root_dist.tolist()
    return feature_dict
