task PrepareTokenizerCorpus
    > corpus_dir
    :: data_path=@
    :: len=@
    :: repo=@
    :: languages=@
{
    python $repo/tokenizer_scripts/sample_tokenizer_data.py \
        --data $data_path \
        --language $languages \
        --len $len \
        --output_dir $corpus_dir
}

task TrainTokenizer
    <  data=corpus_dir@PrepareTokenizerCorpus
    >  tokenizer_dir
    :: repo=@
{   
    python $repo/tokenizer_scripts/train_tokenizer.py \
        --data_path $data \
        --vocab_size $vocab_size \
        --output_dir $tokenizer_dir
}

task AnalyseTokenizer
    <  tokenizer_dir=tokenizer_dir@TrainTokenizer
    >  analysis_results
    :: repo=@
    :: datasets=@
    :: languages=@
{   
    python $repo/tokenizer_scripts/analyse_tokenizer.py \
        --tokenizer_dir $tokenizer_dir \
        --datasets $datasets \
        --languages $languages
}