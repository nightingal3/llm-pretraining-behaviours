#!/bin/bash

# Arithmetic
for url in \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/single_digit_three_ops.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/two_digit_addition.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/two_digit_subtraction.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/two_digit_multiplication.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/three_digit_addition.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/three_digit_subtraction.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/four_digit_addition.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/four_digit_subtraction.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/five_digit_addition.jsonl" \
"https://raw.githubusercontent.com/openai/gpt-3/master/data/five_digit_subtraction.jsonl"
do
  curl $url >> contaminant.txt
done

# Asdiv
curl https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/master/dataset/ASDiv.xml >> contaminant.txt

# GSM8k
curl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl >> contaminant.txt

# MathQA
mkdir mathqa && cd $_
wget https://math-qa.github.io/math-QA/data/MathQA.zip
cat test.json >> ../contaminant.txt
cd ..
rm -rf mathqa

# minerva_math
mkdir minerva && cd $_
wget https://people.eecs.berkeley.edu/~hendrycks/MATH.tar
tar -xvf MATH.tar
rm MATH.tar
for dir in MATH/test/*; do
  for f in $dir/*; do
    cat $f >> ../contaminant.txt
  done
done
cd ..
rm -rf minerva

# LogQA 2.0
curl https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/Test.txt >> contaminant.txt
curl https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master/zh_test.txt >> contaminant.txt

# Fld
curl "https://datasets-server.huggingface.co/rows?dataset=hitachi-nlp%2FFLD.v2&config=default&split=test" >> contaminant.txt
curl "https://datasets-server.huggingface.co/rows?dataset=hitachi-nlp%2FFLD.v2&config=star&split=test" >> contaminant.txt

# Proscript
wget https://storage.googleapis.com/ai2-mosaic-public/projects/proscript/proscript_v1a.zip
jar -xvf proscript_v1a.zip
cat proscript_v1a/test.jsonl >> contaminant.txt
rm -rf proscript_v1a
rm proscript_v1a.zip

# Scrolls
mkdir scrolls && cd $_
for url in \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/gov_report.zip" \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/summ_screen_fd.zip" \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/qmsum.zip" \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/narrative_qa.zip" \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/qasper.zip" \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/quality.zip" \
"https://huggingface.co/datasets/tau/scrolls/resolve/main/contract_nli.zip"
do
  wget $url
done
for file in *; do
  jar -xvf $file
  rm $file
done
for dir in *; do
  cat $dir/test.jsonl >> ../contaminant.txt
done
cd ..
rm -rf scrolls

# Lambada
mkdir lambada && cd $_
curl -O https://zenodo.org/records/2630551/files/lambada-dataset.tar.gz?download=1
tar -xvf lambada-dataset.tar.gz
cat lambada_control_test_data_plain_text.txt >> ../contaminant.txt
cat lambada_test_plain_text.txt >> ../contaminant.txt
cd ..
rm -rf lambada

# Propara
curl https://raw.githubusercontent.com/allenai/propara/master/data/emnlp18/grids.v1.test.json >> contaminant.txt

# Entity tracking in LMs
mkdir entity && cd $_
wget https://github.com/sebschu/entity-tracking-lms/raw/main/data/boxes-dataset-v1.zip
unzip -P iamnotaLM boxes-dataset-v1.zip
for dir in boxes-dataset-v1/*; do
  if [ -d "$dir" ]; then
    cat $dir/test-t5.jsonl >> ../contaminant.txt
  fi
done
cd ..
rm -rf entity