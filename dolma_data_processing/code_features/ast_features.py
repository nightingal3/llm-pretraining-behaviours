import argparse
import json
from tree_sitter import Parser, Node, Language, Query
from tree_sitter_languages import get_language, get_parser
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io import TextIOWrapper


def _trunc_str(s: str, maxLen: int = 10):
    s = s.replace("\n", "\\n")
    if len(s) < maxLen:
        return s
    else:
        return s[: maxLen - 3] + "..."


# May be useful later if we want to query with more specificity - for now we'll just query by type
def _field_name(node: Node):
    parent = node.parent
    if not parent:
        return "None"
    cursor = parent.walk()
    child = cursor.goto_first_child()
    while child != None and cursor.node != node:
        cursor.goto_next_sibling()
    if cursor.node == node and cursor.field_name != None:
        return cursor.field_name
    return "None"


def _write_node_with_content(node: Node, output_file: str, level: int = 0):
    indent = "  " * level
    with open(output_file, "a") as f:
        f.write(
            f"{indent}Node id : {node.id}, Node type: {node.type}, Node sexp: {node.sexp()}, Node text: {_trunc_str(node.text.decode())}\n"
        )

    for child in node.named_children:
        _write_node_with_content(child, output_file, level + 1)


def _traverse_get_depth(
    node: Node,
    word_depths: dict,
    output_file: TextIOWrapper,
    depth: int = 0,
):
    indent = "   " * depth
    word_depths[node] = depth
    f.write(f"{indent}{node.id} ({_trunc_str(str(node.text), 10)}): {depth}\n")
    for child in node.named_children:
        _traverse_get_depth(child, word_depths, output_file, depth + 1)


def _query_get_distances(
    root: Node, lang: Language, paths: dict
) -> dict[str : set[Node]]:
    if lang.name not in paths:
        raise ValueError(f"Language {lang.name} not found in the paths dictionary.")

    # Builds queries of structure (outer: (inner) @query-name)
    def _build_nested_queries(
        outer_patterns: list[str], inner_patterns: list[str], capture_name: str
    ) -> list[Query]:
        queries = []
        for outer in outer_patterns:
            for inner in inner_patterns:
                try:
                    queries.append(lang.query(f"({outer} ({inner}) @{capture_name})"))
                except Exception as e:
                    print(e)
                    continue
        return queries

    # query.captures returns List[Tuple[Node, str]], but we only care about the nodes.
    # This covnerts List[Tuple[Node, str]] to List[Node].
    def _get_nodes(node_str_tuples: list[tuple[Node, str]]) -> set[Node]:
        return set(map(lambda x: x[0], node_str_tuples))

    captures: dict[str : set[Node]] = {
        "func_defs": set(),
        "func_calls": set(),
        "var_defs": set(),
        "var_usgs": set(),
    }

    # Make a separate query for each way of defining a function
    func_def_queries: list[Query] = _build_nested_queries(
        paths[lang.name]["FuncIndicators"],
        paths[lang.name]["FuncNameIndicators"],
        "func-name",
    )
    for query in func_def_queries:
        captures["func_defs"] = captures["func_defs"] | _get_nodes(query.captures(root))

    # Make a separate query for each way of calling a function
    func_call_queries: list[Query] = _build_nested_queries(
        paths[lang.name]["CallIndicators"],
        paths[lang.name]["CallNameIndicators"],
        "func-call",
    )
    for query in func_call_queries:
        captures["func_calls"] = captures["func_calls"] | _get_nodes(
            query.captures(root)
        )

    # Make a separate query for each way of defining a variable
    var_def_queries: list[Query] = _build_nested_queries(
        paths[lang.name]["VarDefIndicators"],
        paths[lang.name]["VarDefNameIndicators"],
        "var-def",
    )
    for query in var_def_queries:
        captures["var_defs"] = captures["var_defs"] | _get_nodes(query.captures(root))

    # Make a separate query for each way of referencing a variable
    # First, gather all the variable names captured above
    # extract the text of each var_def node, then remove duplicates via list(set(...))
    var_names = list(
        set(list(map(lambda node: node.text.decode(), captures["var_defs"])))
    )
    # Query for all instances of var_usg_indicator matching a var_name in var_names
    var_usg_queries: list[Query] = []
    for var_name in var_names:
        for var_usg_indidcator in paths[lang.name]["VarUsgIndicators"]:
            var_usg_queries.append(
                lang.query(
                    f"""({var_usg_indidcator}) @var-usg
                                              (#eq? @var-usg "{var_name}")"""
                )
            )
    # Assumption: All identifiers which are not function definitions/calls or variable definitions are variable usages
    exclude: set[Node] = captures["func_defs"].union(
        captures["func_calls"], captures["var_defs"]
    )
    for query in var_usg_queries:
        captures["var_usgs"] = captures["var_usgs"] | (
            _get_nodes(query.captures(root)) - exclude
        )

    return captures


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
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

    try:
        lang = get_language(args.lang)
    except Exception as e:
        print(f"Error occurred while getting language: {e}")

    with open(args.input_file, "r") as file:
        code = file.read()
    tree = parser.parse(bytes(code, "utf-8"))
    with open(args.output_file, "w") as f:
        f.write("")
    _write_node_with_content(tree.root_node, args.output_file)
    with open(args.output_file, "a") as f:
        word_depths = {}
        f.write(f"\n" + "-" * 50 + "\nWord depths:\n")
        _traverse_get_depth(node=tree.root_node, word_depths=word_depths, output_file=f)
        feature_dict["word_depths"] = word_depths
        feature_dict["tree_depth"] = max(word_depths.values())
        f.write("-" * 50 + f"\nTree depth: {feature_dict['tree_depth']}\n")
        feature_dict["words"] = word_depths.keys()
        f.write("-" * 50 + f"\nNum words: {len(feature_dict['words'])}\n")

        with open("ast_feature_paths.json", "r") as paths_file:
            paths = json.load(paths_file)
            captures: dict[str : list[Node]] = _query_get_distances(
                tree.root_node, lang, paths
            )
            f.write("-" * 50 + "\nDefs/Usgs:\n")
            for key in captures.keys():
                f.write(f"   {key}:\n")
                for node in captures[key]:
                    f.write(
                        f"      {_trunc_str(node.text.decode())} ({node.start_point}:{node.end_point})\n"
                    )