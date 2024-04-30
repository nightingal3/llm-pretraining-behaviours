import pytest
import os
from tree_sitter_languages import get_language, get_parser
from ast_features import *


def test_get_features_working_code():
    """
    Tests the features returned by get_features
    """

    input_code = """
def test_func(x):
    return x
a = 1
a += test_func(a)
    """
    node_depths = [0, 1, 2, 2, 3, 2, 3, 4, 1, 2, 3, 3, 1, 2, 3, 3, 4, 4, 5]
    tree_depths = [0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5]
    dists_to_defs = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        14,
        None,
        None,
        None,
        None,
        None,
        None,
        5,
        None,
        29,
        None,
        20,
    ]
    node_types = [
        108,
        146,
        1,
        147,
        1,
        161,
        125,
        1,
        122,
        197,
        1,
        93,
        122,
        198,
        1,
        205,
        1,
        158,
        1,
    ]
    num_nodes_input = [19] * 19

    features = get_features(input_code, get_language("python"), get_parser("python"))

    assert node_depths == features["node_depth"], "node depths do not match"
    assert tree_depths == features["tree_depth"], "tree depths do not match"
    print(features["dist_to_def"])
    assert (
        dists_to_defs == features["dist_to_def"]
    ), "distances between definitions and usages do not match"
    assert node_types == features["node_type"], "node types do not match"
    assert (
        num_nodes_input == features["num_nodes_input"]
    ), "total number of nodes do not mach"


def test_get_features_broken_code():
    input_code = """
# this is a comment
This code is broken!
    """
    node_depths = [0, 1, 1, 2, 2, 3, 3, 2]
    tree_depths = [0, 1, 1, 2, 2, 3, 3, 3]
    dists_to_defs = [None, None, None, None, None, None, None, None]
    node_types = [108, 99, 65535, 1, 194, 1, 1, 65535]
    num_nodes_input = [8] * 8

    features = get_features(input_code, get_language("python"), get_parser("python"))

    assert node_depths == features["node_depth"], "node depths do not match"
    assert tree_depths == features["tree_depth"], "tree depths do not match"
    print(features["dist_to_def"])
    assert (
        dists_to_defs == features["dist_to_def"]
    ), "distances between definitions and usages do not match"
    assert node_types == features["node_type"], "node types do not match"
    assert (
        num_nodes_input == features["num_nodes_input"]
    ), "total number of nodes do not mach"


test_code_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ast_testing/test_code"
)
output_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ast_testing/output"
)


@pytest.mark.parametrize("lang_name", os.listdir(test_code_dir))
def test_other_langs(lang_name):
    # get features for the other languages
    # this only checks that they run without errors
    # see ast_testing/outputs for the parsing results
    parser = get_parser(lang_name)
    lang = get_language(lang_name)
    with open(f"{test_code_dir}/{lang_name}", "r") as input_file:
        code = input_file.read()
    tree = parser.parse(bytes(code, "utf-8"))
    feature_dict = get_features(code, lang, parser)

    output_string = tree_to_string(tree.root_node, 0)
    for key in feature_dict:
        output_string += "\n" + "-" * 50 + "\n"
        output_string += f"\n{key}: {feature_dict[key]}\n"

    with open(f"{output_dir}/{lang_name}", "w") as output_file:
        output_file.write(output_string)


def test_get_languages():
    """
    Tests that tree-sitter-languages can load all the languages we're covering
    """

    test_code_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ast_testing/test_code"
    )

    for lang_name in os.listdir(test_code_dir):
        language = get_language(lang_name)
        parser = get_parser(lang_name)
