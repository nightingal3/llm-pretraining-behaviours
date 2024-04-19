import os

langs = os.listdir("test_code")
for lang in langs:
    os.system(
        f"python3 ../ast_features.py --lang={lang} --input_file=test_code/{lang} --output_file=output/{lang}"
    )
