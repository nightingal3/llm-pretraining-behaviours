import argparse
from tree_sitter import Parser, Node
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
    node: Node, source_code: str, output_file: str, level: int = 0
):
    """
    Recursively write the node's type, its start and end positions, and its text content.

    Args:
        node (tree_sitter.Node): the node to print
        source_code (str): the source code
        level (int): the depth of the node (used for indentation)
    """
    indent = "  " * level
    node_text = source_code[node.start_byte : node.end_byte]
    with open(output_file, "a") as f:
        f.write(
            f"{indent}Node id : {node.id}, Node type: {node.type}, start: {node.start_point}, end: {node.end_point}, \n{indent}text: '{node_text}'\n"
        )

    for child in node.children:
        write_node_with_content(child, source_code, output_file, level + 1)


def _traverse_get_depth(
    node: Node,
    source_code: str,
    word_depths: dict,
    output_file: TextIOWrapper,
    depth: int = 0,
):
    indent = "   " * depth
    word_depths[node] = depth
    node_text = source_code[node.start_byte : node.end_byte]
    f.write(f"{indent}{node.id} ({trunc_str(node_text, 10)}): {depth}\n")
    for child in node.children:
        _traverse_get_depth(child, source_code, word_depths, output_file, depth + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="The language to parse (if known)")
    parser.add_argument("--input_file", type=str, help="The file to parse")
    parser.add_argument("--output_file", type=str, help="Output file")
    args = parser.parse_args()

    avail_languages = load_languages()
    if args.lang not in avail_languages:
        print(f"Language {args.lang} not available")
        exit(1)
    parser = Parser()
    parser.set_language(avail_languages[args.lang])

    with open(args.input_file, "r") as file:
        code = file.read()
    tree = parser.parse(bytes(code, "utf-8"))
    with open(args.output_file, "w") as f:
        f.write("")
    write_node_with_content(tree.root_node, code, args.output_file)
    with open(args.output_file, "a") as f:
        f.write(f"\n" + "-" * 50 + "\nWord depths:\n")
        word_depths = {}
        _traverse_get_depth(tree.root_node, code, word_depths, f)
