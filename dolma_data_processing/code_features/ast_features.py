import argparse
from tree_sitter import Parser, Node
from tree_sitter_languages import get_language, get_parser
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io import TextIOWrapper
from tree_sitter_load import load_languages


def trunc_str(s: str, maxLen: int):
    s = s.replace("\n", "\\n")
    if len(s) < maxLen:
        return s
    else:
        return s[: maxLen - 3] + "..."


def write_node_with_content(
    node: Node, output_file: str, level: int = 0
):
    """
    Recursively write the node's type, its start and end positions, and its text content.

    Args:
        node (tree_sitter.Node): the node to print
        source_code (str): the source code
        level (int): the depth of the node (used for indentation)
    """
    indent = "  " * level
    with open(output_file, "a") as f:
        f.write(
            # f"{indent}Node id : {node.id}, Node grammar name: {node.grammar_name}, Node text: {node.text}\n"
            f"{indent}Node id : {node.id}, Node type: {node.type}, Node text: {node.text}\n"
        )

    for child in node.children:
        write_node_with_content(child, output_file, level + 1)


def _traverse_get_depth(
    node: Node,
    word_depths: dict,
    output_file: TextIOWrapper,
    depth: int = 0,
):
    indent = "   " * depth
    word_depths[node] = depth
    f.write(f"{indent}{node.id} ({trunc_str(str(node.text), 10)}): {depth}\n")
    for child in node.children:
        _traverse_get_depth(child, word_depths, output_file, depth + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="The language to parse (if known)")
    parser.add_argument("--input_file", type=str, help="The file to parse")
    parser.add_argument("--output_file", type=str, help="Output file")
    args = parser.parse_args()

    feature_dict = {
        "word_depths": {},
        "tree_depth": 0,
        "words": [],
    }

    try:
        parser = get_parser(args.lang)
    except Exception as e:
        print(f"Error occurred while creating parser: {e}")

    with open(args.input_file, "r") as file:
        code = file.read()
    tree = parser.parse(bytes(code, "utf-8"))
    with open(args.output_file, "w") as f:
        f.write("")
    write_node_with_content(tree.root_node, args.output_file)
    with open(args.output_file, "a") as f:
        word_depths = {}
        f.write(f"\n" + "-" * 50 + "\nWord depths:\n")
        _traverse_get_depth(node=tree.root_node, word_depths=word_depths, output_file=f)
        feature_dict["word_depths"] = word_depths
        feature_dict["tree_depth"] = max(word_depths.values())
        f.write("-" * 50 + f"\nTree depth: {feature_dict['tree_depth']}\n")
        feature_dict["words"] = word_depths.keys()
        f.write("-" * 50 + f"\nNum words: {len(feature_dict['words'])}")

