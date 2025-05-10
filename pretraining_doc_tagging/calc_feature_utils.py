"""Simple features:
## Sequence level
# 1. Number of characters
# 2. Number of tokens
# 3. Lexical diversity (avg rank of ngrams within the sequence)
## Word/token level 
# TODO: ask Graham about this as well?
# 3. Number of unique tokens
# 4. Ngram frequency of this token
# 5. Position in sequence
# 6. Number of times this token appears in the sequence"""



# for keyword type feats
keywords = {
    "question_words":      r"\b(How|What|Why|When|Where|Who|Which|Whose)\b",
    "imperative_verbs":    r"\b(Do|Make|Consider|Take|Use|Ensure|Check|Build|Apply|Run|Create|Find|Go|Try|Turn|Start|Stop|Put|Keep|Leave|Get|Move)\b",
    "conjunctions":        r"\b(and|but|or|so|because|although|however|therefore|yet)\b",
    "instruction_words":   r"(Question:|Answer:|Instruction:|User:|Assistant:|Q:|A:)",
    "numbers":             r"\b\d+\b|\b\d+\.\d+\b|\b\d+%\b",
}

def get_keyword_ratios(text: str) -> dict:
    total_chars = len(text) or 1
    counts = {k: len(re.findall(p, text, flags=re.IGNORECASE)) for k, p in keywords.items()}
    return {f"{k}_ratio": 100_000 * counts[k] / total_chars for k in keywords}

def get_num_tokens(tokenized_input: list[int]) -> int:
    return len(tokenized_input)


def get_num_chars(string_input: str) -> int:
    return len(string_input)


def get_lexical_diversity(
    tokenized_input: list[int], freq_dict: dict[tuple[int], int]
) -> float:
    """
    Lexical diversity is the average rank of the ngrams in the sequence
    """
    return sum([freq_dict[(token)] for token in tokenized_input]) / len(tokenized_input)


def get_num_unique_tokens(tokenized_input: list[int]) -> int:
    return len(set(tokenized_input))


def get_position_in_sequence(tokenized_input: list[int]) -> list[int]:
    return list(range(len(tokenized_input)))


def get_num_times_token_appears(tokenized_input: list[int], token: int) -> int:
    return tokenized_input.count(token)
