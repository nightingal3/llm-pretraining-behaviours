Writing these temporary notes to document where I left, off, please delete off the branch when you're done! -E

### Basic things we want to tag:
**"Constituency"**
We can basically refer to the features in `dolma_data_processing/calc_parse_feature_utils.py` and use all of them (replace POS with node type, and ignore 'sentence' features)

**"Dependency"**
1. Dependency between a function definition and each invocation (e.g. distance between def *someFunc*(...) .... and each call of *someFunc*)
    - You can just create one edge between the function invocation and the first token of the function name/definition
2. Dependency between a variable definition and its usage (similar to the above)


### Tree-sitter
We discussed some basic options [here](https://docs.google.com/document/d/1KrQTx_d3naBheZt1C6fy7YvhHP-aT9nlBBvVap-IMt4/edit?usp=sharing), but tree-sitter seemed to be the most promising after our initial investigation. The starter code is for tree-sitter, but it something else happens to work better, we can consider using it instead.

Tree-sitter returns an AST (you can use the starter code to print out some ASTs on simple code to see how it's structured). We want to use some features from the AST, but then store the final data features in the same format as for natural language (see `dolma_data_processing/calc_parse_feature_utils.py` for return types)

### Possible tree-sitter detail

In `download_tree_sitter_parsers.sh`, I'm using the recomended pip installation [here](dolma_data_processing/calc_parse_feature_utils.py), but it seems like most languages aren't available through pip. We may have to use the "build from source" method on that page (would need to manually download each language parser though)


