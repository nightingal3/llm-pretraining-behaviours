import json
import os
import sys
from typing import Callable, Union
import argparse
import warnings
import logging
import time
from collections import defaultdict

import pyspark
from pyspark.sql.types import (
    StructType,
    StructField,
    ArrayType,
    IntegerType,
    StringType,
)
import pyspark.sql.functions as F
import pandas as pd
from tqdm import tqdm
import stanza
from transformers import AutoTokenizer
import zstandard as zstd

from tree_sitter import Parser, Language
from tree_sitter_languages import get_language, get_parser

from calc_feature_utils import *
import calc_parse_feature_utils
from code_features.ast_features import get_features as get_code_features
from hf_classifier import TextClassifierHf

feature_dict_schema_classifier = StructType(
    [
        StructField("raw_score", ArrayType(IntegerType())),
        StructField("int_score", ArrayType(IntegerType())),
    ]
)


def register_classifier(name: str, model_name: str):
    """Add a classifier to the feature registry"""
    classifier = TextClassifierHf(model_name)

    def classifier_fn(texts):
        return classifier.predict_batch([texts])[0]

    feature_registry[name] = {
        "tagging_fn": classifier_fn,
        "need_tokenize": False,
        "need_parse": False,
        "need_code_parse": False,
        "need_raw_text": True,
        "dtype": feature_dict_schema_classifier,
    }


register_classifier("edu_classifier", "HuggingFaceTB/fineweb-edu-classifier")

# Existing feature definitions
feature_dict_schema_const = StructType(
    [
        StructField("words", ArrayType(StringType())),
        StructField("const_word_depth", ArrayType(IntegerType())),
        StructField("const_tree_depth", ArrayType(IntegerType())),
        StructField("upos_label", ArrayType(StringType())),
        StructField("xpos_label", ArrayType(StringType())),
        StructField("num_words_sentence", ArrayType(IntegerType())),
        StructField("num_words_input", ArrayType(IntegerType())),
        StructField("num_sentences_input", ArrayType(IntegerType())),
    ]
)

feature_dict_schema_deps = StructType(
    [
        StructField("dist_to_head", ArrayType(IntegerType())),
        StructField("dist_to_root", ArrayType(IntegerType())),
    ]
)

# New schema for code features
feature_dict_schema_code = StructType(
    [
        StructField("node_depth", ArrayType(IntegerType())),
        StructField("tree_depth", ArrayType(IntegerType())),
        StructField("dist_to_def", ArrayType(IntegerType())),
        StructField("node_type", ArrayType(IntegerType())),
        StructField("num_nodes_input", ArrayType(IntegerType())),
    ]
)

# Updated feature registry
feature_registry = {
    "num_tokens": {
        "tagging_fn": get_num_tokens,
        "need_tokenize": True,
        "need_parse": False,
        "dtype": IntegerType(),
    },
    "char_len": {
        "tagging_fn": get_num_chars,
        "need_tokenize": False,
        "need_parse": False,
        "dtype": IntegerType(),
    },
    "unique_tokens": {
        "tagging_fn": get_num_unique_tokens,
        "need_tokenize": True,
        "need_parse": False,
        "dtype": IntegerType(),
    },
    "seq_ind_tok": {
        "tagging_fn": get_position_in_sequence,
        "need_tokenize": True,
        "need_parse": False,
        "dtype": ArrayType(IntegerType()),
    },
    "dep_parse": {
        "tagging_fn": calc_parse_feature_utils.get_dep_parse_features,
        "need_tokenize": False,
        "need_parse": True,
        "dtype": feature_dict_schema_deps,
    },
    "const_parse": {
        "tagging_fn": calc_parse_feature_utils.get_const_parse_features,
        "need_tokenize": False,
        "need_parse": True,
        "dtype": feature_dict_schema_const,
    },
    "code_features": {
        "tagging_fn": get_code_features,
        "need_tokenize": False,
        "need_parse": False,
        "need_code_parse": True,
        "dtype": feature_dict_schema_code,
    },
}

# Global variables
stanza_pipeline = defaultdict(lambda: None)
stanza_langdetect_pipeline = None
tree_sitter_parsers = {}


def get_tree_sitter_parser(lang_name):
    if lang_name not in tree_sitter_parsers:
        try:
            parser = get_parser(lang_name)
            lang = get_language(lang_name)
            tree_sitter_parsers[lang_name] = (parser, lang)
        except Exception as e:
            logging.error(f"Error creating parser for {lang_name}: {e}")
            return None, None
    return tree_sitter_parsers[lang_name]


def feat_dict_to_row(feat_dict: dict) -> pyspark.sql.Row:
    return pyspark.sql.Row(**feat_dict)


def parse_features_udf_wrapper(
    feature_fn: Callable, stanza_args: str, schema: StructType
) -> pyspark.sql.functions.udf:
    def parse_features_udf(text: str, langcode: str = "en") -> pyspark.sql.Row:
        global stanza_pipeline
        if stanza_pipeline[langcode] is None:
            try:
                stanza_pipeline[langcode] = stanza.Pipeline(
                    lang=langcode, processors=stanza_args
                )
            except:  # language not supported?
                return feat_dict_to_row({k: [] for k in schema.fieldNames()})

        language_pipeline = stanza_pipeline[langcode]
        return feat_dict_to_row(feature_fn(text, language_pipeline))

    return pyspark.sql.functions.udf(parse_features_udf, schema)


def code_features_udf_wrapper(schema: StructType) -> pyspark.sql.functions.udf:
    def code_features_udf(code: str, lang_name: str) -> pyspark.sql.Row:
        parser, lang = get_tree_sitter_parser(lang_name)
        if parser is None or lang is None:
            return feat_dict_to_row({k: [] for k in schema.fieldNames()})

        feature_dict = get_features(code, lang, parser)
        return feat_dict_to_row(feature_dict)

    return pyspark.sql.functions.udf(code_features_udf, schema)


def detect_lang(text: str) -> str:
    global stanza_langdetect_pipeline
    if stanza_langdetect_pipeline is None:
        stanza_langdetect_pipeline = stanza.Pipeline(
            lang="multilingual", processors="langid"
        )
    return stanza_langdetect_pipeline(text).lang[:2]


def process_with_pandas(
    feature: str,
    df: pd.DataFrame,
    tokenizer,
    feature_fn: Callable,
    needs_parse: bool,
    needs_code_parse: bool,
    dtype,
    stanza_args: str = None,
) -> pd.DataFrame:
    """Process features using pandas for smaller files."""
    if "id" not in df.columns:
        logging.info("ID column not found, generating synthetic IDs.")
        df["id"] = range(len(df))

    if "content" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"content": "text"})

    # Handle tokenization
    if feature_registry[feature]["need_tokenize"]:
        logging.info("Tokenizing...")
        df["token_ids"] = df["text"].apply(
            lambda x: tokenizer(x, add_special_tokens=False)["input_ids"]
        )
    else:
        df["token_ids"] = df["text"]

    # Process features
    if needs_parse:
        logging.info("Detecting languages...")

        def get_lang(text):
            return detect_lang(text)

        df["lang"] = df["text"].apply(get_lang)

        logging.info("Calculating parse features...")

        def process_row(row):
            try:
                parser = stanza_pipeline[row["lang"]]
                if parser is None:
                    # initialize stanza pipeline
                    try:
                        stanza_pipeline[row["lang"]] = stanza.Pipeline(
                            lang=row["lang"], processors=stanza_args
                        )
                    except ValueError as e:
                        # Language not supported - return empty features
                        logging.warning(f"Unsupported language {row['lang']}, skipping")
                        if feature == "content_function_ratio":
                            return 0.0
                        return {k: [] for k in feature_dict_schema_const.fieldNames()} if feature == "const_parse" else \
                            {k: [] for k in feature_dict_schema_deps.fieldNames()}
                
                # Use existing pipeline
                return feature_fn(row["text"], stanza_pipeline[row["lang"]])
            except Exception as e:
                logging.warning(f"Error processing row with language {row['lang']}: {str(e)}")
                if feature == "content_function_ratio":
                    return 0.0
                return {k: [] for k in feature_dict_schema_const.fieldNames()} if feature == "const_parse" else \
                    {k: [] for k in feature_dict_schema_deps.fieldNames()}
                    

        features = [process_row(row) for _, row in tqdm(df.iterrows())]
    elif needs_code_parse:
        logging.info("Calculating code features...")

        def process_code_row(row):
            parser, lang = get_tree_sitter_parser(row["lang"])
            if parser is None or lang is None:
                return {k: [] for k in dtype.fieldNames()}
            return get_features(row["text"], lang, parser)

        features = [process_code_row(row) for _, row in tqdm(df.iterrows())]
    else:
        logging.info("Calculating features...")
        features = [feature_fn(tokens) for tokens in tqdm(df["token_ids"])]

    # Convert features to DataFrame
    feature_df = pd.DataFrame({"id": df["id"], feature: features})

    return feature_df


def main(feature: str, input_filepath: str, output_filepath: str, limit: int = None):
    SIZE_THRESHOLD_GB = 20

    feature_fn = feature_registry[feature]["tagging_fn"]
    do_tokenize = feature_registry[feature]["need_tokenize"]
    dtype = feature_registry[feature]["dtype"]
    needs_parse = feature_registry[feature]["need_parse"]
    needs_code_parse = feature_registry[feature].get("need_code_parse", False)
    stanza_args = (
        "tokenize,pos,constituency"
        if feature == "const_parse"
        else "tokenize,pos,depparse"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "/data/models/huggingface/meta-llama/Llama-2-70b-hf/"
    )
    output_filepath = (
        output_filepath
        if output_filepath is not None and output_filepath.endswith(".parquet")
        else f"{feature}.parquet"
    )

    input_file_size = os.path.getsize(input_filepath) / (1024**3)
    if input_file_size <= SIZE_THRESHOLD_GB:
        logging.info(f"File size is < {SIZE_THRESHOLD_GB}GB, using pandas to process")
        if input_filepath.endswith(".jsonl"):
            df = pd.read_json(input_filepath, lines=True, nrows=limit)
        elif input_filepath.endswith(".jsonl.zst"):
            df = read_jsonl_zst_pandas(input_filepath, limit)
        else:
            df = pd.read_parquet(input_filepath, nrows=limit)

        # Process with pandas
        feature_df = process_with_pandas(
            feature=feature,
            df=df,
            tokenizer=tokenizer,
            feature_fn=feature_fn,
            needs_parse=needs_parse,
            needs_code_parse=needs_code_parse,
            dtype=dtype,
            stanza_args=stanza_args,
        )
<<<<<<< Updated upstream

        # print some stats about the feature
        logging.info(f"Feature {feature} stats:")
        stats = feature_df[feature].apply(pd.Series).describe().T
        stats = stats[["mean", "std", "min", "max"]]
        stats = stats.applymap(lambda x: f"{x:.2f}")
        logging.info(stats)
=======
        if not feature in ["const_parse", "dep_parse", "code_features"]:
            # print some stats about the feature
            logging.info(f"Feature {feature} stats:")
            stats = feature_df[feature].apply(pd.Series).describe().T
            stats = stats[["mean", "std", "min", "max"]]
            stats = stats.applymap(lambda x: f"{x:.2f}")
            logging.info(stats)
>>>>>>> Stashed changes
        # Save output
        feature_df.to_parquet(output_filepath)
        logging.info(f"Saved feature {feature} to {output_filepath}")
    else:
        spark = (
            pyspark.sql.SparkSession.builder.master("local[*]")
            .config("spark.driver.memory", "50G")
            .config("spark.executor.memory", "40G")
            .getOrCreate()
        )
        broadcast_tokenizer = spark.sparkContext.broadcast(tokenizer)

        def tokenize(text: str) -> list[int]:
            return broadcast_tokenizer.value(text, add_special_tokens=False)[
                "input_ids"
            ]

        tokenize_udf = pyspark.sql.functions.udf(
            tokenize, pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType())
        )

        logging.info(f"Reading {input_filepath}...")
        if input_filepath.endswith(".jsonl"):
            df = spark.read.json(input_filepath)
        elif input_filepath.endswith(".jsonl.zst"):
            df = read_jsonl_zst(spark, input_filepath)
        else:
            df = spark.read.parquet(input_filepath)

        if "content" in df.columns and "text" not in df.columns:
            df = df.withColumnRenamed("content", "text")

        if limit is not None:
            df = df.limit(limit)
        df.cache()

        if "id" not in df.columns:
            logging.info("ID column not found, generating synthetic IDs.")
            df = df.withColumn("id", F.monotonically_increasing_id())

        if do_tokenize:
            logging.info("Tokenizing...")
            df = df.withColumn("token_ids", tokenize_udf("text"))
        else:
            df = df.withColumn("token_ids", F.col("text"))

        if needs_parse:
            feature_udf = parse_features_udf_wrapper(feature_fn, stanza_args, dtype)
            detect_lang_udf = pyspark.sql.functions.udf(
                detect_lang, pyspark.sql.types.StringType()
            )
            df = df.withColumn("lang", detect_lang_udf("text"))
        elif needs_code_parse:
            feature_udf = code_features_udf_wrapper(dtype)
        else:
            feature_udf = pyspark.sql.functions.udf(feature_fn, dtype)

        logging.info("Calculating feature...")

        if needs_parse:
            df = df.withColumn(feature, feature_udf("text", "lang"))
        elif needs_code_parse:
            df = df.withColumn(feature, feature_udf("text", "lang"))
        else:
            df = df.withColumn(feature, feature_udf("token_ids"))

        feature_df = df.select("id", feature)
        if not feature in ["const_parse", "dep_parse", "code_features"]:
            # print some stats about the feature
            logging.info(f"Feature {feature} stats:")
            stats = feature_df.select(
                F.format_number(F.avg(F.col(feature)), 3).alias("mean"),
                F.format_number(F.stddev(F.col(feature)), 2).alias("stddev"),
                F.min(F.col(feature)).alias("min"),
                F.max(F.col(feature)).alias("max"),
            )

        stats.show()

        feature_df.write.parquet(output_filepath, mode="overwrite")
        logging.info(f"Saved feature {feature} to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature", choices=list(feature_registry.keys()), required=True
    )
    parser.add_argument(
        "--input", help="Input file (arrow, jsonl, jsonl.zst)", type=str, required=True
    )
    parser.add_argument("--output", help="Output file (arrow)", type=str)
    parser.add_argument("--limit", help="Limit the number of rows to process", type=int)
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    start_time = time.time()
    main(args.feature, args.input, args.output, args.limit)
    end_time = time.time()
    time_in_min = (end_time - start_time) / 60
    logging.info(f"Total execution time: {time_in_min} min")
