find /data/tir/projects/tir6/general/mengyan3/dolma-features -type d -name "*.parquet" | while read dir; do 
    new_name=$(echo "$dir" | sed -E 's/_(c4|common-crawl|peS2o|stack-code|gutenberg-books)(_|\.parquet)/_dolma-\1\2/g')
    if [ "$dir" != "$new_name" ]; then
        mv "$dir" "$new_name"
    fi
done
