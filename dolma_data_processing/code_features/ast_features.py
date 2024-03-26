import argparse
from tree_sitter import Parser, Node
from tree_sitter_load import load_languages


def print_node_with_content(node: Node, source_code: str, level: int = 0):
    """
    Recursively print the node's type, its start and end positions, and its text content.

    Args:
        node (tree_sitter.Node): the node to print
        source_code (str): the source code
        level (int): the depth of the node (used for indentation)
    """
    indent = "  " * level
    node_text = source_code[node.start_byte : node.end_byte]
    print(
        f"{indent}Node type: {node.type}, start: {node.start_point}, end: {node.end_point}, text: '{node_text}'"
    )

    for child in node.children:
        print_node_with_content(child, source_code, level + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="The language to parse (if known)")
    parser.add_argument("--input_file", type=str, help="The file to parse")
    args = parser.parse_args()

    avail_languages = load_languages()
    if args.lang not in avail_languages:
        print(f"Language {args.lang} not available")
        exit(1)
    parser = Parser()
    parser.set_language(avail_languages[args.lang])

    code = "import argparse\nif __name__ == '__main__':\nprint('Hello, world!')"
    tree = parser.parse(bytes(code, "utf-8"))
    print_node_with_content(tree.root_node, code)
