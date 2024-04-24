import argparse
import json
from tree_sitter import Parser, Node, Language, Query
from tree_sitter_languages import get_language, get_parser
from typing import Union
import os
import sys
import warnings
import bisect

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from io import TextIOWrapper


def _trunc_str(s: str, maxLen: int = 20):
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
            f"{indent}Node id : {node.id}, Node type: {node.type}, Node text: {_trunc_str(node.text.decode())}\n"
        )

    for child in node.named_children:
        _write_node_with_content(child, output_file, level + 1)


def _traverse_get_depths(
    node: Node,
    node_depths: dict[Node, int],
    node_depth: int,
) -> int:
    node_depths[node] = node_depth
    for child in node.named_children:
        _traverse_get_depths(child, node_depths, node_depth + 1)


# Calls tree-sitter queries to find nodes of function/variable definitions/usages
def _query_get_funcs_and_vars(
    root: Node, lang: Language, paths: dict
) -> dict[str, set[Node]]:
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

    captures: dict[str, set[Node]] = {
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


# Takes in the output of _query_get_funcs_and_vars and calculates distances from usages to definitions
def _get_distances(
    captures: dict[str, set[Node]],
) -> tuple[dict[Node, int], dict[Node, int]]:
    # Maps each function/variable usage to its closest previous matching definition
    closest: dict[Node, Node] = {}

    # Organize defs into dicts where key is node text
    func_defs: dict[str, list[Node]] = {}
    var_defs: dict[str, list[Node]] = {}
    for func_def in captures["func_defs"]:
        if func_def.text.decode() in func_defs:
            func_defs[func_def.text.decode()].append(func_def)
        else:
            func_defs[func_def.text.decode()] = [func_def]
    for var_def in captures["var_defs"]:
        if var_def.text.decode() in var_defs:
            var_defs[var_def.text.decode()].append(var_def)
        else:
            var_defs[var_def.text.decode()] = [var_def]

    func_defs = {
        name: sorted(func_defs[name], key=lambda node: node.end_byte)
        for name in func_defs.keys()
    }
    var_defs = {
        name: sorted(var_defs[name], key=lambda node: node.end_byte)
        for name in var_defs.keys()
    }

    for func_call in captures["func_calls"]:
        if func_call.text.decode() in func_defs.keys():
            matching_defs = func_defs[func_call.text.decode()]
            matching_defs_end_bytes = list(
                map(lambda node: node.end_byte, matching_defs)
            )
            index = bisect.bisect_left(matching_defs_end_bytes, func_call.start_byte)
            closest[func_call] = (
                matching_defs[max(index - 1, 0)]
                if index < len(matching_defs)
                else matching_defs[-1]
            )
    for var_usg in captures["var_usgs"]:
        if var_usg.text.decode() in var_defs.keys():
            matching_defs = var_defs[var_usg.text.decode()]
            matching_defs_end_bytes = list(
                map(lambda node: node.end_byte, matching_defs)
            )
            index = bisect.bisect_left(matching_defs_end_bytes, var_usg.start_byte)
            closest[var_usg] = (
                matching_defs[max(index - 1, 0)]
                if index < len(matching_defs)
                else matching_defs[-1]
            )

    func_distances: dict[Node, int] = {
        node: node.start_byte - closest[node].end_byte
        for node in captures["func_calls"]
    }
    var_distances: dict[Node, int] = {
        node: node.start_byte - closest[node].end_byte for node in captures["var_usgs"]
    }
    return (func_distances, var_distances)


def get_features(
    input_code: str, lang: Language, parser: Parser
) -> dict[str, list[Union[str, int, float, None]]]:
    """
    Takes as input a chunk of code and tree-sitter lang/parser
    Returns a feature dict, where values are lists features per AST node.

    features:
    node_depth: the depth of the node in the AST
    tree_depth: the depth of the AST overall
    dist_to_def: the byte distance from this node to its function/variable definition, or None if not applicable
    num_nodes_input: the total number of nodes in the AST
    """
    feature_dict = {
        "node_depth": [],
        "tree_depth": [],
        "dist_to_def": [],
    }
    tree = parser.parse(bytes(input_code, "utf-8"))
    # maps node -> (depth of node, depth of tree at that node)
    node_depths: dict[Node, int] = {}
    _traverse_get_depths(tree.root_node, node_depths, 0)

    with open("ast_feature_paths.json", "r") as paths_file:
        paths = json.load(paths_file)
    funcs_and_vars_dict = _query_get_funcs_and_vars(tree.root_node, lang, paths)
    (func_distances, var_distances) = _get_distances(funcs_and_vars_dict)

    # tree_depth = "maximum depth of tree up to this point"
    tree_depth = 0
    for node in node_depths:
        feature_dict["node_depth"].append(node_depths[node])
        tree_depth = max(tree_depth, feature_dict["node_depth"][-1])
        feature_dict["tree_depth"].append(tree_depth)
        if node in func_distances:
            feature_dict["dist_to_def"].append(func_distances[node])
        elif node in var_distances:
            feature_dict["dist_to_def"].append(var_distances[node])
        else:
            feature_dict["dist_to_def"].append(None)

    return feature_dict


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, help="The language to parse (if known)")
    parser.add_argument("--input_file", type=str, help="The file to parse")
    parser.add_argument("--output_file", type=str, help="Output file")
    args = parser.parse_args()

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
        f.write("\n" + "-" * 50 + "\n")
        feature_dict = get_features(code, lang, parser)
        for key in feature_dict:
            f.write("\n" + key + ": " + str(feature_dict[key]))
