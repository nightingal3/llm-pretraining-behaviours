import argparse
import multiprocessing
import pyarrow
import pyspark
import pyspark.sql.functions as F
import pandas as pd
from tqdm import tqdm
import logging
from transformers import AutoTokenizer

from calc_feature_utils import *

feature_registry = {
    "num_tokens": {
        "tagging_fn": get_num_tokens,
        "need_tokenize": True,
        "dtype": pyspark.sql.types.IntegerType(),
    },
    "char_len": {
        "tagging_fn": get_num_chars,
        "need_tokenize": False,
        "dtype": pyspark.sql.types.IntegerType(),
    },
    "lexical_diversity": {
        "tagging_fn": get_lexical_diversity,
        "need_tokenize": True,
        "dtype": pyspark.sql.types.FloatType(),
    },
    "unique_tokens": {
        "tagging_fn": get_num_unique_tokens,
        "need_tokenize": True,
        "dtype": pyspark.sql.types.IntegerType(),
    },
    "seq_ind_tok": {
        "tagging_fn": get_position_in_sequence,
        "need_tokenize": True,
        "dtype": pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType()),
    }
}

def main(feature: str, input_filepath: str, output_filepath: str):
    feature_fn = feature_registry[feature]["tagging_fn"]
    do_tokenize = feature_registry[feature]["need_tokenize"]
    dtype = feature_registry[feature]["dtype"]
    tokenizer = AutoTokenizer.from_pretrained("/data/datasets/models/huggingface/meta-llama/Llama-2-70b-hf/")
    
    spark = pyspark.sql.SparkSession.builder.master("local[*]").config("spark.driver.memory", "30G").getOrCreate()
    broadcast_tokenizer = spark.sparkContext.broadcast(tokenizer)

    def tokenize(text: str) -> List[int]:
        return broadcast_tokenizer.value(text, add_special_tokens=False)["input_ids"]

    tokenize_udf = pyspark.sql.functions.udf(tokenize, pyspark.sql.types.ArrayType(pyspark.sql.types.IntegerType()))
    feature_udf = pyspark.sql.functions.udf(feature_fn, dtype)
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
    
    logging.info("Calculating feature...")
    df = df.withColumn(feature, feature_udf("token_ids"))
    feature_df = df.select(feature)
    feature_df.write.parquet(output_filepath, mode="overwrite")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", choices=["entropy", "char_len", "num_tokens", "lexical_diversity", "unique_tokens", "seq_ind_tok"])
    parser.add_argument("--input", help="Input file (arrow)", type=str)
    parser.add_argument("--output", help="Output file (arrow)", type=str)
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    main(args.feature, args.input, args.output)