task PrepareTokenizerCorpus
    > corpus_dir
    :: data_path_de=@
    :: data_path_en=@
    :: data_path_es=@
    :: data_path_fr=@
    :: data_path_it=@
    :: data_path_ko=@
    :: data_path_nl=@
    :: data_path_pl=@
    :: data_path_pt=@
    :: data_path_ru=@
    :: data_path_sv=@
    :: data_path_zh=@
    :: n_words=@
    :: repo=@
    :: languages=$languages
{
    mkdir corpus_dir
    python $repo/tokenizer_scripts/prepare_tokenizer_corpus.py \
        --data_path_de $data_path_de \
        --data_path_en $data_path_en \
        --data_path_es $data_path_es \
        --data_path_fr $data_path_fr \
        --data_path_it $data_path_it \
        --data_path_ko $data_path_ko \
        --data_path_nl $data_path_nl \
        --data_path_pl $data_path_pl \
        --data_path_pt $data_path_pt \
        --data_path_ru $data_path_ru \
        --data_path_sv $data_path_sv \
        --data_path_zh $data_path_zh \
        --languages $languages \
        --max_words $n_words \
        --output_dir $corpus_dir
}

task TrainTokenizer
    <  data=$corpus_dir@PrepareTokenizerCorpus
    >  tokenizer_dir
    :: repo=@
    :: vocab_size=@
    :: extra_tokens=@
{   
    mkdir tokenizer_dir
    python $repo/tokenizer_scripts/train_tokenizer.py \
        --data_path $data \
        --vocab_size $vocab_size \
        --output_dir $tokenizer_dir \
        --extra_tokens $extra_tokens
}

task AnalyseTokenizer
    <  tokenizer_dir=$tokenizer_dir@TrainTokenizer
    >  analysis_results
    :: repo=@
    :: eval_languages=@
    :: dataset=@
{   
    python $repo/tokenizer_scripts/analyse_tokenizer.py \
        --tokenizer_dir $tokenizer_dir \
        --dataset $dataset \
        --languages $eval_languages
}


plan Train {
    reach TrainTokenizer via (NWords: 100000000) * (VocabSize: 64000 96000 128000 160000 192000)
}

plan Analyse {
    reach AnalyseTokenizer via (NWords: 100000000) * (VocabSize: 64000 96000 128000 160000 192000)
}
