task PrepareTokenizerCorpus
    > corpus_dir
    :: data_path=@
    :: n_words=@
    :: repo=@
    :: languages=$languages
{
    mkdir corpus_dir
    python $repo/tokenizer_scripts/prepare_tokenizer_corpus.py \
        --data $data_path \
        --languages $languages \
        --max_words $n_words \
        --output_dir $corpus_dir
}

task TrainTokenizer
    <  data=$corpus_dir@PrepareTokenizerCorpus
    >  tokenizer_dir
    :: repo=@
    :: vocab_size=@
{   
    mkdir tokenizer_dir
    python $repo/tokenizer_scripts/train_tokenizer.py \
        --data_path $data \
        --vocab_size $vocab_size \
        --output_dir $tokenizer_dir
}

task AnalyseTokenizer
    <  tokenizer_dir=$tokenizer_dir@TrainTokenizer
    >  analysis_results
    :: repo=@
    :: languages=@
    :: dataset=@
{   
    python $repo/tokenizer_scripts/analyse_tokenizer.py \
        --tokenizer_dir $tokenizer_dir \
        --dataset $dataset \
        --languages $languages
}


plan Train {
    reach TrainTokenizer via (NWords: 50000000 100000000 1000000000) * (VocabSize: 50000 75000 100000 125000 150000 175000 200000)
}

plan Analyse {
    reach AnalyseTokenizer via (NWords: 100000 500000 1000000 5000000 10000000 50000000 100000000) * (VocabSize: 50000 75000 100000 125000 150000 175000 200000)
}
