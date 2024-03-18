import os

# Specify the directory path you want to check
directory_path = '/data/tir/projects/tir7/user_data/mchen5/dolma_100B/c4'

num_files = len(os.listdir(directory_path))

print(f"Number of files in '{directory_path}': {num_files}")
