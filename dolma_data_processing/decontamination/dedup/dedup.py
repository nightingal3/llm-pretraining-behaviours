import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd
import sys
import os
import multiprocessing
import traceback
import logging

sys.path.append(
    "/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/lm-evaluation-harness"
)
from lm_eval.decontamination.janitor import Janitor

os.environ["NUMEXPR_MAX_THREADS"] = "256"
import numexpr as ne

# Make janitor, register contaminant
with open(
    "/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma_data_processing/decontamination/dedup/contaminant.txt",
    "r",
) as file:
    contaminant: str = file.read()
janitor = Janitor()
janitor.register_contaminant(contaminant)
logging.info("Created janitor, registered contaminant")


def decontaminate(df: pd.DataFrame) -> pd.DataFrame:
    df["num_contaminated"] = 0
    df["thrown"] = False

    num_thrown = 0
    for index, row in df.iterrows():
        try:
            (cleaned, num_contaminated) = janitor.clean_python(row["text"])
            df.at[index, "num_contaminated"] = num_contaminated
            if num_contaminated != 0:
                df.at[index, "text"] = "".join(cleaned)
        except:
            df.at[index, "thrown"] = True
            num_thrown += 1

    return df


contamination_indices = 0

logging.info(f"Using {multiprocessing.cpu_count()} CPUs")


# Add exception handling to multiprocessing Process
class Process(multiprocessing.Process):

    def __init__(self, *args, **kwargs):
        multiprocessing.Process.__init__(self, *args, **kwargs)
        self._pconn, self._cconn = multiprocessing.Pipe()
        self._exception = None

    def run(self):
        try:
            multiprocessing.Process.run(self)
            self._cconn.send(None)
        except Exception as e:
            tb = traceback.format_exc()
            self._cconn.send((e, tb))

    @property
    def exception(self):
        if self._pconn.poll():
            self._exception = self._pconn.recv()
        return self._exception


# Deduplicate the file at this path and saves the output to dolma_100B_deduped
def process_file(file_path, directory_name, file_name):
    logging.info(f"Processing {file_path}; process id {os.getpid()} \n")
    global contamination_indices
    df: pd.DataFrame = pq.read_table(file_path).to_pandas()
    df = decontaminate(df)
    contamination_indices += df["num_contaminated"].sum()
    df.to_parquet(
        f"/data/tir/projects/tir7/user_data/mchen5/dolma_100B_deduped/{directory_name}/{file_name}"
    )
    logging.info(f"Finished writing deduped {file_name}")


# Start a new process for each file, so we deduplicate fully in parallel
def process_directory(directory_path, directory_name):
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            p = multiprocessing.Process(
                target=process_file, args=(file_path, directory_name, file_name)
            )
            p.start()


def main():
    global contamination_indices
    base_dir = "/data/tir/projects/tir7/user_data/mchen5/dolma_100B"
    directory_processes = []
    for directory_name in os.listdir(base_dir):
        directory_path = os.path.join(base_dir, directory_name)
        if os.path.isdir(directory_path):
            p = multiprocessing.Process(
                target=process_directory, args=(directory_path, directory_name)
            )
            directory_processes.append(p)
            p.start()
    for p in directory_processes:
        p.join()
    logging.info("Finished decontamination")
    logging.info(f"{contamination_indices} total contamination indices")


if __name__ == "__main__":
    main()
