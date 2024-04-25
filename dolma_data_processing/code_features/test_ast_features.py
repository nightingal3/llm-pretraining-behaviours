import os
import warnings
from ast_features import *
from tree_sitter_languages import get_language, get_parser


def _trunc_str(s: str, maxLen: int = 20):
    s = s.replace("\n", "\\n")
    if len(s) < maxLen:
        return s
    else:
        return s[: maxLen - 3] + "..."


def _tree_to_string(node: Node, level: int = 0) -> str:
    indent = "  " * level
    string = (
        f"{indent}Node type: {node.type}, Node.text: {_trunc_str(node.text.decode())}\n"
    )
    for child in node.named_children:
        string += _tree_to_string(child, level + 1)
    return string


def test_ast_features():
    """
    Runs get_features in ast_features on the test code files in ast_testing/test_code
    Output tree/features for each file sent to ast_testing/output
    """
    warnings.filterwarnings("ignore")

    test_code_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ast_testing/test_code"
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ast_testing/output"
    )
    for lang_name in os.listdir(test_code_dir):
        print(f"Testing ast_features on {lang_name}...")

        try:
            parser = get_parser(lang_name)
        except Exception as e:
            print(f"Error ocurred while getting parser for {lang_name}: {e}")

        try:
            lang = get_language(lang_name)
        except Exception as e:
            print(f"Error ocurred while getting language for {lang_name}: {e}")

        with open(f"{test_code_dir}/{lang_name}", "r") as input_file:
            code = input_file.read()
        tree = parser.parse(bytes(code, "utf-8"))
        feature_dict = get_features(code, lang, parser)

        output_string = _tree_to_string(tree.root_node, 0)
        for key in feature_dict:
            output_string += "\n" + "-" * 50 + "\n"
            output_string += f"\n{key}: {feature_dict[key]}\n"

        with open(f"{output_dir}/{lang_name}", "w") as output_file:
            output_file.write(output_string)


if __name__ == "__main__":
    test_ast_features()
