from typing import List, Dict, Tuple, Any, Union, Optional, NewType

### Simple features:
## Sequence level
# 1. Number of characters
# 2. Number of tokens
# 3. Lexical diversity (avg rank of ngrams within the sequence)
## Word/token level 
# TODO: ask Graham about this as well?
# 3. Number of unique tokens
# 4. Ngram frequency of this token
# 5. Position in sequence
# 6. Number of times this token appears in the sequence


tokenized_input = NewType("tokenized_input", List[int])
freq_dict = NewType("freq_dict", Dict[Tuple[int], int])

def get_num_tokens(tokenized_input: tokenized_input) -> int:
    return len(tokenized_input)

def get_num_chars(string_input: str) -> int:
    return len(string_input)

def get_lexical_diversity(tokenized_input: tokenized_input, freq_dict: freq_dict) -> float:
    """
    Lexical diversity is the average rank of the ngrams in the sequence
    """
    return sum([freq_dict[(token)] for token in tokenized_input]) / len(tokenized_input)

def get_num_unique_tokens(tokenized_input: tokenized_input) -> int:
    return len(set(tokenized_input))

def get_one_ngram_freq(key: Tuple[int], freq_dict: freq_dict) -> int:
    return freq_dict[key]

def get_position_in_sequence(tokenized_input: tokenized_input, token: int) -> int:
    return tokenized_input.index(token)

def get_num_times_token_appears(tokenized_input: tokenized_input, token: int) -> int:
    return tokenized_input.count(token)
