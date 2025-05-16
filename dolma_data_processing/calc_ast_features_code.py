import tree_sitter
from tree_sitter_languages import get_language, get_parser
import os


def init_parser(lang: str) -> tree_sitter.Parser:
    """
    Initialize the tree-sitter parser for the given language
    @lang: str: the language to initialize the parser for
    @return: tree_sitter.Parser: the parser
    """
    language = get_language(lang)
    parser = get_parser(language)
    return parser


def code_langdetect(code: str) -> str:
    """
    Detect the language of the given code
    @code: str: the code to detect the language of
    @return: str: the detected language
    """
    parser = tree_sitter.Parser()
    for lang in [
        "python",
        "java",
        "cpp",
        "javascript",
        "go",
        "ruby",
        "php",
        "rust",
        "c_sharp",
        "c",
    ]:
        lang = tree_sitter.Language(
            os.path.join(
                os.path.dirname(__file__), f"tree_sitter/tree-sitter-{lang}.so"
            )
        )
        parser.set_language(lang)
        if parser.parse(bytes(code, "utf-8")).root_node is not None:
            return lang
    return None


def parse_code(parser: tree_sitter.Parser, code: str) -> tree_sitter.Tree:
    """
    Parse the given code using the given parser
    @parser: tree_sitter.Parser: the parser to use
    @code: str: the code to parse
    @return: tree_sitter.Tree: the parsed tree
    """
    tree = parser.parse(bytes(code, "utf-8"))
    return tree
