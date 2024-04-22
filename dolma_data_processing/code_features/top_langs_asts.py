import os

langs = os.listdir(
    "/data/tir/projects/tir7/user_data/mchen5/llm-pretraining-behaviours/dolma_data_processing/code_features/ast_testing/test_code"
)
for lang in langs:
    print(lang + ":")
    os.system(
        f"python3 ast_features.py --lang={lang} --input_file=ast_testing/test_code/{lang} --output_file=ast_testing/output/{lang}"
    )
