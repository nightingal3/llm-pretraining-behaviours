import pytest
import stanza
from calc_parse_feature_utils import *


def test_const_parse_features():
    """
    Tests count and depth of constituency tree calculations
    """
    pipeline = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    input_text = "This is a sentence missing a period This is the first test. \nThis has a newline before it."
    features = get_const_parse_features(input_text, pipeline)
    num_words_input = [20] * 20
    num_sentences_input = [3] * 20
    sent_word_depths = [[3, 3, 5, 5, 5, 6, 6],
                        [3, 3, 4, 4, 4, 2],
                        [3, 3, 4, 4, 4, 5, 2]]
    sent_tree_depths = [6, 4, 5]
    const_word_depth = []
    const_tree_depth = []
    num_words_sentence = []
    for i, sent in enumerate(sent_word_depths):
        const_word_depth += sent
        const_tree_depth += [sent_tree_depths[i]] * len(sent_word_depths[i])
        num_words_sentence += [len(sent_word_depths[i])] * len(sent_word_depths[i])
    assert const_word_depth == features["const_word_depth"], "word depths do not match"
    assert const_tree_depth == features["const_tree_depth"], "tree depths do not match"
    assert num_words_sentence == features["num_words_sentence"], "number of words per sentence do not match"
    assert num_words_input == features["num_words_input"], "number of total words do not match"
    assert num_sentences_input == features["num_sentences_input"], "number of sentences do not match"


def test_dep_parse_features():
    """
    Test depth of dependency tree calculations
    """
    pipeline = stanza.Pipeline(lang='en', processors="tokenize,pos,lemma,depparse")
    input_text = "The quick brown fox jumped over the lazy dog. I am writing a unit test."
    features = get_dep_parse_features(input_text, pipeline)
    sent_dist_to_head = [[3, 2, 1, 1, 0, 3, 2, 1, 4, 5],
                         [2, 1, 0, 2, 1, 3, 4]]
    sent_dist_to_root = [[4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
                         [2, 1, 0, 1, 2, 3, 4]]
    dist_to_head = []
    dist_to_root = []
    for i in range(len(sent_dist_to_head)):
        dist_to_head += sent_dist_to_head[i]
        dist_to_root += sent_dist_to_root[i]
    assert dist_to_head == features["dist_to_head"], "distances to head do not match"
    assert dist_to_root == features["dist_to_root"], "distances to root do not match"
