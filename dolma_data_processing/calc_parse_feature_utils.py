import stanza
from typing import List, Dict, Tuple, Union, Optional, NewType

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
# 1. Depth of word
# 2. Depth of tree
# 3. Distance of word from immediate head

FeatureDict = NewType("FeatureDict", Dict[str, List[Union[str, int, float]]])


def get_const_parse_features(input_text: str) -> FeatureDict:
    """
    Takes as input a chunk of text (paragraph, document)
    Returns a feature dict, where values are lists features per word
    """
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

    feature_dict = FeatureDict({
        "const_word_depth": [],
        "const_tree_depth": [],
        "upos_label": [],
        "xpos_label": [],
        "num_words_sentence": [],
        "num_words_input": [],
        "num_sentences_input": [],
    })
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    processed_text = pipeline(input_text)
    num_words_input = 0
    for sentence in processed_text.sentences:
        num_words_input += len(sentence.words)
        for word in sentence:
            feature_dict.num_sentences_input.append(len(processed_text.sentences))
            feature_dict.num_words_sentence.append(len(sentence.words))
            feature_dict.upos_label.append(word.upos)
            feature_dict.xpos_label.append(word.xpos)
        word_depths, max_depth = get_depth_features(sentence.constituency)
        feature_dict.const_word_depth += word_depths
        feature_dict.const_tree_depth += max_depth
    feature_dict.num_words_input = [num_words_input] * num_words_input
    return feature_dict
