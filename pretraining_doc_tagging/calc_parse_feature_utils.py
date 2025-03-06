import numpy as np
import stanza
from typing import Union, Any

"""Parse-based features (word + sentence level)
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
# 2. Distance of word to root """


def get_content_function_ratio(input_text: str, pipeline: stanza.Pipeline) -> float:
    """Compute ratio of content words to total words"""
    try:
        processed_text = pipeline(input_text)
        pos_tags = [
            word.upos
            for sentence in processed_text.sentences
            for word in sentence.words
        ]
        content_tags = {"NOUN", "VERB", "ADJ", "ADV"}
        if not pos_tags:
            return 0.0
        content_words = sum(1 for tag in pos_tags if tag in content_tags)
        return content_words / len(pos_tags)
    except Exception as e:
        return 0.0


def _traverse_get_depth(
    tree,
    word_depths: list[int],
    depth: int,
):
    if len(tree.children) == 0:
        word_depths.append(depth)
        return
    depth += 1
    subtrees = tree.children
    for subtree in subtrees:
        _traverse_get_depth(subtree, word_depths, depth)


def _get_depth_features(
    tree: Any,
) -> tuple[list[int], list[int]]:
    word_depths = []
    tree = tree.children[0]
    _traverse_get_depth(tree, word_depths, 0)
    max_depth = [max(word_depths)] * len(word_depths)
    return word_depths, max_depth


def get_const_parse_features(
    input_text: str,
    pipeline: stanza.Pipeline,
) -> dict[str, list[Union[str, int, float]]]:
    """
    Takes as input a chunk of text (paragraph, document) and stanza pipeline
    Lang for the pipeline should be 'en', processors are 'tokenize,pos,constituency'
    Returns a feature dict, where values are lists features per word.

    features:
    words: a list of words in the text
    num_sentences_input: the sentence count in the input
    num_words_sentence: the word count per sentence
    upos_label: the universal POS label for the word
    xpos_label: the language-specific POS label for the word
    const_word_depth: the depth of the word in the constituency tree
    const_tree_depth: the depth of the constituency tree overall
    num_words_input: the total number of words
    """
    feature_dict = {
        "words": [],
        "const_word_depth": [],
        "const_tree_depth": [],
        "upos_label": [],
        "xpos_label": [],
        "num_words_sentence": [],
        "num_words_input": [],
        "num_sentences_input": [],
    }

    processed_text = pipeline(input_text)
    num_words_input = 0
    for sentence in processed_text.sentences:
        num_words_input += len(sentence.words)
        for word in sentence.words:
            feature_dict["words"].append(word.text)
            feature_dict["num_sentences_input"].append(len(processed_text.sentences))
            feature_dict["num_words_sentence"].append(len(sentence.words))
            feature_dict["upos_label"].append(word.upos)
            feature_dict["xpos_label"].append(word.xpos)
        word_depths, max_depth = _get_depth_features(sentence.constituency)
        feature_dict["const_word_depth"] += word_depths
        feature_dict["const_tree_depth"] += max_depth
    feature_dict["num_words_input"] = [num_words_input] * num_words_input
    return feature_dict


def get_dep_parse_features(
    input_text: str,
    pipeline: stanza.Pipeline,
) -> dict[str, list[Union[str, int, float]]]:
    """
    Takes as input a chunk of text (paragraph, document) and stanza pipeline
    Lang for the pipeline should be 'en', processors are 'tokenize,pos,lemma,depparse'
    Returns a feature dict, where values are lists features per word
    """
    feature_dict = {"words": [], "dist_to_head": [], "dist_to_root": []}
    processed_text = pipeline(input_text)
    for sentence in processed_text.sentences:
        root_dist = np.zeros(len(sentence.words), dtype=int)
        head_dist = []
        root = -1
        for i, word in enumerate(sentence.words):
            feature_dict["words"].append(word.text)
            root_dist[i] = word.id
            if word.head == 0:
                head_dist.append(0)
            else:
                head_dist.append(abs(word.id - word.head))
            if word.head == 0:
                root = word.id
        root_dist -= root
        root_dist = np.abs(root_dist)
        feature_dict["dist_to_head"] += head_dist
        feature_dict["dist_to_root"] += root_dist.tolist()
    return feature_dict
