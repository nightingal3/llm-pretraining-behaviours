This directory contains the code for calculating code features, as well as some useful testing scripts/test results.

# Code Features

Features are calculated from abstract syntax trees using the `tree-sitter` library. `ast_features.py` contains the code for calculating:
- Depth of each AST node
- Total depth of the AST at each node (i.e. the maximum depth of an earlier node in the tree)
- Distance (in bytes) from each function/variable usage to its definition
- Grammar type id of each node
- Total number of nodes in the AST

### Different Grammars for Different Languages

Different languages have different `tree-sitter` grammars, so their ASTs have different structure. Thus, for each language, we have to manually input the 
AST paths to function/variable definitions/usages. These paths are in `ast_feature_paths.json`:
- For each function definition, the function name is found at one of the paths in `FuncIndicators`/`FuncNameIndicators`
- For each function call, the function mame is found at one of the paths in `CallIndicators`/`CallNameIndicators`
- For each variable definition, the variable name is found at one of the paths in `VarDefIndicators`/`VarDefNameIndicators`
- For each variable usage, the variable name is found at one of the paths in `VarUsgIndicators`
    - To prevent false positives, we filter to nodes which are not function definitions/calls or variable definitions, and whose names are found in variable definitions.

The AST paths are included for the top ~80% of languages in the stack-code domain. See `stack_analysis` for the language distribution. (Note: `stack_breakdown.csv` includes "languages" like html and markdown, which are not included in `stack_breakdown_real.csv`.)

## Testing

`test_ast_features.py` has unit tests for `ast_features.py` - run with `pytest`.

`run_ast_features.py` runs `ast_features.py`'s `get_features` function on the files in `ast_testing`/`test_code`. The output ASTs and features are in `ast_testing/output`.

### Dependencies

Pip install:
- [tree-sitter](https://github.com/tree-sitter/py-tree-sitter)
- [tree-sitter-languages](https://github.com/grantjenks/py-tree-sitter-languages)
