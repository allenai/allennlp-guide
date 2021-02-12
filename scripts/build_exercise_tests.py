import glob
from dataclasses import dataclass
import re
from typing import Optional, Iterator


OUTPUT_DIR = "_exercises_test/"

CODEBLOCK_RE = re.compile(r"<codeblock source=\"([^\"]+)\"( setup=\"([^\"]+)\")?>")


@dataclass
class CodeExercise:
    source: str
    setup: Optional[str] = None


def find_code_exercises() -> Iterator[CodeExercise]:
    # sort to ensure the order is deterministic.
    for filepath in sorted(glob.glob("./chapters/**/*.md")):
        with open(filepath) as f:
            text = f.read()
        for (source, _, setup) in CODEBLOCK_RE.findall(text):
            yield CodeExercise(
                source="exercises/" + source + ".py",
                setup="exercises/" + setup + ".py" if setup else None,
            )


def main():
    for i, code_exercise in enumerate(find_code_exercises()):
        test_filename = OUTPUT_DIR + f"exercise{i}.py"
        with open(test_filename, "w") as test_file:
            if code_exercise.setup is not None:
                test_file.write(f"# setup: {code_exercise.setup}\n\n")
                with open(code_exercise.setup) as setup_file:
                    for line in setup_file:
                        test_file.write(line)
                test_file.write("\n\n")
            test_file.write(f"# source: {code_exercise.source}\n\n")
            with open(code_exercise.source) as source_file:
                for line in source_file:
                    test_file.write(line)


if __name__ == "__main__":
    main()
