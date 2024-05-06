#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH --partition=babel-shared-long
#SBATCH --cpus-per-task=30
#SBATCH --mem=50G
#SBATCH --job-name=get_dataset_features
#SBATCH --output=get_dataset_features-%a.out

# script to calculate and update aggregate feature statistics to metadata json files

set -euxo pipefail

conda activate towerllm-env

# TODO: update this to be done in parallel across jsons in an array job
metadata_file='metadata/dataset_metadata/dolma.json'
dataset=$(basename $metadata_file .json)
feature="num_tokens"
# TODO: update this to be a more permanent dir, ideally shared
output_feature_dir="scratch/lindiat"
mkdir -p $TEMP_DIR

num_domains=$(jq -r '.domains | length' $metadata_file)
end=(num_domains-1)

for ((i=0;i<=end;i++));
do
    file_pointer=$(jq ".domains[$i].file_pointer" $metadata_file)
    domain_name=$(jq ".domains[$i].name" $metadata_file)
    if [ "$file_pointer" == "null" ]
    then
        echo "no data file pointer present"
    else
        # assuming that the file pointer points to a dir of jsonl files
        if [ ! -d $file_pointer ]; then
            echo "$file_pointer does not exist"
        else
            num_files=$(find $file_pointer -type f -name '*jsonl' | wc -l)
            for $input_file in $file_pointer/*.jsonl
                do
                    output_file=${output_feature_dir}/${dataset}/${domain_name}/${feature}/${(basename $input_file)}
                    # TODO: write script to get features with jsonl files as input
                    python get_dataset_features.py \
                        --feature $feature \
                        --input $input_file \
                        --output $output_file
                done | tqdm --total $num_files
            # TODO: write a script to aggregate over the dataset
            # and update the current metadata json (update_metadata_features.py)
            feature_dir=${output_feature_dir}/${dataset}/${domain_name}/${feature}
            python update_metadata_features.py \
                --feature_dir $feature_dir \
                --metadata_file $metadata_file
            echo "${metadata_file} updated to include aggregate ${feature} statistics"
        fi
    fi
done
