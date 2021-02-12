import json
from typing import List
from allennlp.common import Registrable, Params


class Count(Registrable):
    def __init__(self, count: int):
        self.count = count

    @classmethod
    def from_list_of_ints(cls, int_list: List[int]):
        return cls(len(int_list))

    @classmethod
    def from_list_of_strings(cls, str_list: List[str]):
        return cls(len(str_list))

    @classmethod
    def from_string_length(cls, string: str):
        return cls(len(string))


# We can't use the @Count.register() decorator before the Count class is defined,
# so we have to manually call the decorator here, below.  If we were using a
# subclass of Count, we could have just used the @Count.register() decorator
# multiple times.
Count.register("default")(Count)
Count.register("from_list_of_ints", constructor="from_list_of_ints")(Count)
Count.register("from_list_of_strings", constructor="from_list_of_strings")(Count)
Count.register("from_string_length", constructor="from_string_length")(Count)
Count.default_implementation = "default"


param_str = """{"count": 23}"""
count = Count.from_params(Params(json.loads(param_str)))
print(f"Count 1: {count.count}")

param_str = """{"type": "from_list_of_ints", "int_list": [1, 2, 3]}"""
count = Count.from_params(Params(json.loads(param_str)))
print(f"Count 2: {count.count}")

param_str = """{"type": "from_list_of_strings", "str_list": ["a", "list"]}"""
count = Count.from_params(Params(json.loads(param_str)))
print(f"Count 3: {count.count}")

param_str = """{"type": "from_string_length", "string": "this is a string"}"""
count = Count.from_params(Params(json.loads(param_str)))
print(f"Count 4: {count.count}")
