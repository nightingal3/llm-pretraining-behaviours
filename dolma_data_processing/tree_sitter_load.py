from tree_sitter import Language, Parser
import importlib
import pkg_resources


def load_languages():
    languages = {}

    installed_packages = [pkg.key for pkg in pkg_resources.working_set]
    ts_packages = [
        pkg
        for pkg in installed_packages
        if pkg.startswith("tree-sitter") and pkg != "tree-sitter"
    ]

    for pkg in ts_packages:
        pkg_name = pkg.replace("-", "_")
        print(pkg_name)
        language_name = pkg.split("-")[-1]
        try:
            lang_module = importlib.import_module(pkg_name)
            language = Language(lang_module.language(), language_name)
            languages[language_name] = language
            print(f"Loaded language: {language_name}")
        except:
            print(f"Failed to load language: {language_name}")

    return languages


avail_languages = load_languages()

if __name__ == "__main__":
    parser = Parser()
    code = "int main() { return 0; }"
    lang = "sml"

    if lang in avail_languages:
        parser.set_language(avail_languages[lang])
        tree = parser.parse(bytes(code, "utf-8"))
        print(tree.root_node)
    else:
        print(f"Language {lang} not available")
