import requests
import os
all_files_lst = requests.get("http://128.2.209.71:5000/list-all").json()
all_files_lst = [f for f in all_files_lst if f.endswith(".gz")]

domain_file_lists = {
    "stack-code": [],
    "peS2o": [],
    "common-crawl": [],
    "c4": [],
}
domain_process_counts = {
    "stack-code": 10,
    "peS2o": 5,
    "common-crawl": 10,
    "c4": 5,
}
domain_file_splits = domain_file_lists

for domain in domain_file_lists:
    domain_file_lists[domain] = [f for f in all_files_lst if domain in f]

for domain in domain_file_lists:
    file_ind = 0
    file_list = domain_file_lists[domain]
    num_processes = domain_process_counts[domain]
    domain_file_splits[domain] = [[] for _ in range(num_processes)]
    while file_ind < len(file_list):
        for i in range(num_processes):
            domain_file_splits[domain][i].append(file_list[file_ind])
            file_ind += 1
            if file_ind >= len(file_list):
                break

for domain in domain_file_lists:
    for i, files in enumerate(domain_file_splits[domain]):
        file_path = f"/data/tir/projects/tir7/user_data/mchen5/dolma_file_splits/{domain}/{i}"
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            for file in files:
                f.write(file + '\n')
