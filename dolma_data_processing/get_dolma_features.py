import argparse
import multiprocessing
import pyarrow
import pyspark
from pyspark.sql.types import StructType, StructField, ArrayType, IntegerType, StringType, MapType
import pyspark.sql.functions as F
from typing import Callable
import pandas as pd
from tqdm import tqdm
import logging
import stanza
from transformers import AutoTokenizer
import time
from collections import defaultdict

from calc_feature_utils import *
import calc_parse_feature_utils


feature_dict_schema_const = StructType([
    StructField("words", ArrayType(StringType())),
    StructField("const_word_depth", ArrayType(IntegerType())),
    StructField("const_tree_depth", ArrayType(IntegerType())),
    StructField("upos_label", ArrayType(StringType())),
    StructField("xpos_label", ArrayType(StringType())),
    StructField("num_words_sentence", ArrayType(IntegerType())),
    StructField("num_words_input", ArrayType(IntegerType())),
    StructField("num_sentences_input", ArrayType(IntegerType())),
])

feature_dict_schema_deps = StructType([
    StructField("dist_to_head", ArrayType(IntegerType())),
    StructField("dist_to_root", ArrayType(IntegerType())),
])

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
}

# we probably shouldn't broadcast this, creating global variables instead
stanza_pipeline = defaultdict(lambda: None)
stanza_langdetect_pipeline = None

def feat_dict_to_row(feat_dict: dict) -> pyspark.sql.Row:
    return pyspark.sql.Row(**feat_dict) 

def parse_features_udf_wrapper(feature_fn: Callable, stanza_args: str, schema: StructType) -> pyspark.sql.functions.udf:
    def parse_features_udf(text: str, langcode: str = "en") -> pyspark.sql.Row:
        global stanza_pipeline
        if stanza_pipeline[langcode] is None:
            try:
                stanza_pipeline[langcode] = stanza.Pipeline(lang=langcode, processors=stanza_args)
            except: # language not supported?
                return feat_dict_to_row({k: [] for k in schema.fieldNames()})

        language_pipeline = stanza_pipeline[langcode]
        return feat_dict_to_row(feature_fn(text, language_pipeline))
    
    return pyspark.sql.functions.udf(parse_features_udf, schema)

def detect_lang(text: str) -> str:
    global stanza_langdetect_pipeline
    if stanza_langdetect_pipeline is None:
        stanza_langdetect_pipeline = stanza.Pipeline(lang="multilingual", processors="langid")
    return stanza_langdetect_pipeline(text).lang[:2]

def main(feature: str, input_filepath: str, output_filepath: str):
    feature_fn = feature_registry[feature]["tagging_fn"]
    do_tokenize = feature_registry[feature]["need_tokenize"]
    dtype = feature_registry[feature]["dtype"]
    needs_parse = feature_registry[feature]["need_parse"]
    stanza_args = "tokenize,pos,constituency" if feature == "const_parse" else "tokenize,pos,depparse"

    tokenizer = AutoTokenizer.from_pretrained("/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/")
    output_filepath = output_filepath if output_filepath is not None and output_filepath.endswith(".parquet") else f"{feature}.parquet"
    
    spark = pyspark.sql.SparkSession.builder.master("local[*]").config("spark.driver.memory", "50G").config("spark.executor.memory", "40G").getOrCreate()
    broadcast_tokenizer = spark.sparkContext.broadcast(tokenizer)

    def tokenize(text: str) -> List[int]:
        return broadcast_tokenizer.value(text, add_special_tokens=False)["input_ids"]

    tokenize_udf = pyspark.sql.functions.udf(tokenize, pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType()))

    df = spark.read.parquet(input_filepath)
    # sometimes the text column is called content
    if "content" in df.columns and "text" not in df.columns:
        df = df.withColumnRenamed("content", "text")
    df.cache()
    
    if do_tokenize:
        logging.info("Tokenizing...")
        df = df.withColumn("token_ids", tokenize_udf("text"))
    else:
        df = df.withColumn("token_ids", F.col("text"))
    
    if needs_parse:
        feature_udf = parse_features_udf_wrapper(feature_fn, stanza_args, dtype)
        # detect the languages first
        detect_lang_udf = pyspark.sql.functions.udf(detect_lang, pyspark.sql.types.StringType())
        df = df.withColumn("lang", detect_lang_udf("text"))
        
    else:
        feature_udf = pyspark.sql.functions.udf(feature_fn, dtype)

    logging.info("Calculating feature...")

    if needs_parse:
        df = df.withColumn(feature, feature_udf("text", "lang"))
    else:
        df = df.withColumn(feature, feature_udf("token_ids"))

    feature_df = df.select("id", feature)
    feature_df.write.parquet(output_filepath, mode="overwrite")
    logging.info(f"Saved feature {feature} to {output_filepath}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=list(feature_registry.keys()), required=True)
    parser.add_argument("--input", help="Input file (arrow)", type=str, required=True)
    parser.add_argument("--output", help="Output file (arrow)", type=str)
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    if multiprocessing.cpu_count() < 30:
        raise ValueError("Recommend using at least 30 cores to run this script")
    
    start_time = time.time()
    main(args.feature, args.input, args.output)
    end_time = time.time()
    time_in_min = (end_time - start_time) / 60
    logging.info(f"Total execution time: {time_in_min} min")