import sys
import os
import random
import math
import argparse
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join("..", "nla_semparse")))

from nla_semparse.nla_language import NlaLanguage


class DataGenerator:
    """
    Generator for data points for natural language arithmetic.
    """

    def __init__(self):
        self.language = NlaLanguage()
        self.numbers = [
            {"meaning": "0", "translation": "zero"},
            {"meaning": "1", "translation": "one"},
            {"meaning": "2", "translation": "two"},
            {"meaning": "3", "translation": "three"},
            {"meaning": "4", "translation": "four"},
            {"meaning": "5", "translation": "five"},
            {"meaning": "6", "translation": "six"},
            {"meaning": "7", "translation": "seven"},
            {"meaning": "8", "translation": "eight"},
            {"meaning": "9", "translation": "nine"},
        ]
        # The order below defines precedence (in ascending order).
        self.operators = [
            {"meaning": "subtract", "translation": "minus"},
            {"meaning": "add", "translation": "plus"},
            {"meaning": "multiply", "translation": "times"},
            {"meaning": "divide", "translation": "over"},
        ]

    def generate_expression(
        self, num_operations: int, allowed_operators: List[Dict] = None
    ):
        """
        Generates a single expression that contains the given number of operations.
        """
        if num_operations == 0:
            return random.sample(self.numbers, 1)[0]
        # Expressions will be of the type (OP EXP1 EXP2)
        if allowed_operators is None:
            allowed_operators = self.operators
        operator_index = random.randint(0, len(allowed_operators) - 1)
        operator = allowed_operators[operator_index]
        # Decide how many operators will be in each of EXP1 and EXP2
        random_value = random.random()
        num_operations_for_first = int(num_operations * random_value)
        num_operations_for_second = num_operations - num_operations_for_first - 1
        # The operations in the children will be the same as the operator already sampled, or one of a higher
        # precedence.
        first_argument = self.generate_expression(
            num_operations_for_first, allowed_operators[operator_index:]
        )
        second_argument = self.generate_expression(
            num_operations_for_second, allowed_operators[operator_index:]
        )
        meaning_representation_parts = [
            operator["meaning"],
            first_argument["meaning"],
            second_argument["meaning"],
        ]
        meaning_representation = "(" + " ".join(meaning_representation_parts) + ")"
        return {
            "meaning": meaning_representation,
            "translation": " ".join(
                [
                    first_argument["translation"],
                    operator["translation"],
                    second_argument["translation"],
                ]
            ),
            "denotation": self.language.execute(meaning_representation),
        }

    def generate_data(
        self,
        num_expressions: int,
        min_num_operations: int = 1,
        max_num_operations: int = 10,
        split_data: bool = False,
        train_proportion: float = 0.8,
        test_proportion: float = 0.1,
    ):
        """
        Returns ``num_expressions`` expressions, containing number of operations in the range
        ``(min_num_operations, max_num_operations)``. Optionally, you can also have the data split into
        train, test, and dev sets, ans specify their proportions.
        """
        data: List[Dict[str, Any]] = []
        while len(data) < num_expressions:
            num_operations = random.randint(min_num_operations, max_num_operations)
            try:
                expression = self.generate_expression(num_operations)
                data.append(expression)
            except ZeroDivisionError:
                pass

        if not split_data:
            return {"data": data}
        test_size = math.ceil(test_proportion * num_expressions)
        if train_proportion + test_proportion < 1.0:
            dev_size = math.ceil(
                (1 - (train_proportion + test_proportion)) * num_expressions
            )
        else:
            dev_size = 0
        return {
            "test": data[:test_size],
            "dev": data[test_size : test_size + dev_size],
            "train": data[test_size + dev_size :],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-expressions",
        type=int,
        required=True,
        dest="num_expressions",
        help="Number of expressions to generate",
    )
    parser.add_argument(
        "--min-num-operations", type=int, dest="min_num_operations", default=1
    )
    parser.add_argument(
        "--max-num-operations", type=int, dest="max_num_operations", default=10
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="""Location where output will be written. If splitting data, the name of the split
                                will be appended to the file name""",
    )
    parser.add_argument("--split-data", action="store_true", dest="split")
    parser.add_argument(
        "--train-proportion",
        type=float,
        dest="train_proportion",
        help="How big should the train split be? (Between 0 and 1)",
    )
    parser.add_argument(
        "--test-proportion",
        type=float,
        dest="test_proportion",
        help="""How big should the test split be? (Between 0 and 1). Will also make a dev split
                                if train_proportion + test_proportion < 1""",
    )
    parser.add_argument(
        "--no-meaning",
        action="store_true",
        dest="no_meaning",
        help="Generated data will have denotations instead of meaning",
    )
    args = parser.parse_args()
    if args.no_meaning:
        raise NotImplementedError
    data_generator = DataGenerator()
    data = data_generator.generate_data(
        num_expressions=args.num_expressions,
        min_num_operations=args.min_num_operations,
        max_num_operations=args.max_num_operations,
        split_data=args.split,
        train_proportion=args.train_proportion,
        test_proportion=args.test_proportion,
    )
    if args.split:
        filename_parts = args.output.split(".")
        assert (
            len(filename_parts) == 2
        ), "Cannot decide how to alter the file name. Expected just one ."
        train_file_name = f"{filename_parts[0]}_train.{filename_parts[1]}"
        dev_file_name = f"{filename_parts[0]}_dev.{filename_parts[1]}"
        test_file_name = f"{filename_parts[0]}_test.{filename_parts[1]}"
        with open(train_file_name, "w") as output_file:
            for datum in data["train"]:
                source = datum["translation"]
                target = datum["meaning"].replace("(", "( ").replace(")", " )")
                print(f"{source}\t{target}", file=output_file)
        with open(dev_file_name, "w") as output_file:
            for datum in data["dev"]:
                source = datum["translation"]
                target = datum["meaning"].replace("(", "( ").replace(")", " )")
                print(f"{source}\t{target}", file=output_file)
        with open(test_file_name, "w") as output_file:
            for datum in data["test"]:
                source = datum["translation"]
                target = datum["meaning"].replace("(", "( ").replace(")", " )")
                print(f"{source}\t{target}", file=output_file)
    else:
        with open(args.output, "w") as output_file:
            for datum in data["data"]:
                source = datum["translation"]
                target = datum["meaning"].replace("(", "( ").replace(")", " )")
                print(f"{source}\t{target}", file=output_file)


if __name__ == "__main__":
    main()
