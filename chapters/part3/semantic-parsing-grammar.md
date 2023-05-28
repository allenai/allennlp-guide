---
title: 'Semantic Parsing: Grammar-based Decoding'
description:
  "Here we build on the prior semantic parser chapter and build a better model, using a technique called grammar-based decoding."
type: chapter
---

<exercise id="1" title="Semantic parsing recap">

In the previous chapter we introduced semantic parsing as the task of translating natural language utterances into executable programs.
Recall that we focused on the toy task of Natural Language Arithmetic, that involved translating sequences like
```
three times four plus seven minus five
```
into
```
(subtract (add (multiply 3 4) 7) 5)
```

We took an initial step towards building a semantic parser by viewing the problem as a sequence-to-sequence translation task: the model
encoded the input utterance as a sequence of tokens in natural language (`three`, `times`, ...), and decoded the program as a sequence of elements
in the domain specific programming language (`(`, `subtract`, `(`, ...).

Based on this view, we trained a `Seq2seq` model for the task. Towards the end of the chapter we noted that the model is prone to producing
illegal outputs, particularly when the inputs are long with more than five operators.

We suggested that this issue can be fixed by placing constraints on the output based on the rules of the target programming language. This idea
is based on the following insight: We are using modeling techniques inspired by machine translation, but there is an important difference
between machine translation and semantic parsing: since the target languages in semantic parsing are not natural, they obey certain prespecified
rules without exceptions.

The idea here is to leverage those rules to disallow invalid outputs. Note that the same cannot be done in the case of generating
natural languages, because no matter how complex you make the rules, there are bound to be exceptions due to the nature of human languages.

</exercise>


<exercise id="2" title="Constraints as a grammar">

Let us think about what kind of expressions should be _allowed_ in our language. Here is the complete set of rules:

1. Any digit (`0` - `9`)
2. Bracketed list of an operator followed by two _expressions_, where the operator has to be `add`, `subtract`, `multiply`, or `divide`.

Note that the second rule for valid expressions itself refers to expressions, allowing for recursion in our language.

Expressed as a [context free grammar](https://en.wikipedia.org/wiki/Context-free_grammar), these rules look as follows:

```
@start@ -> int
int -> <int,int:int> int int
<int,int:int> -> add | subtract | multiply | divide
int -> 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
```

If you are not familiar with context free grammars, the idea is that the grammar specifies the process that can be used to generate any
valid expression in the language, and the process is given by the rules in the grammar. The rules contain two kinds of symbols: non-terminals that
can be expanded to make the expressions more complex, and terminals that terminate the expansions. The process of generating any expression starts
out with the special non-terminal `@start@`, and each step in the process involves transforming a non-terminal into the expression on the right of the
`->` in a rule where the non-terminal is on the left side. The process ends when there are no non-terminals left to transform. The selection of rules
used in the process, and the order in which they are used will affect the resulting expressions. In fact, there is a one-to-one mapping between
the sequence of rules used and the resulting expression. Consider reading more about context free grammars
[here](https://en.wikipedia.org/wiki/Context-free_grammar).

The terminals in our NLA grammar are the digits `0` to `9` and the arithmetic operators like `add`. Note that we named the non-terminals
based on the types of the expressions they produce: `int` is a non-terminal which produces integers, and `<int,int:int>` is a function type that
takes two integers as inputs, and produces an integer as the output (the inputs and output types are separated by `:`).

Any valid application of these grammar rules will result in a valid expression in our language. Conversely, this grammar also lets us _parse_
any valid expression in our language into a sequence of these transformation rules. For example, 
the expression `(subtract 3 2)` can be parsed as

```
@start@ -> int
int -> <int,int:int> int int
<int,int:int> -> subtract
int -> 3
int -> 2
```

Note that this sequence of rules is a linearization of the following tree. Specifically, it results from the depth-first traversal of the tree.
The terminals are shown in bold.

<img src="/part3/semantic-parsing/nla_parse_tree_simple.svg" alt="Parse tree for the expression (subtract 3 2)" width="300"/>

Similarly, the expression we saw at the beginning of this chapter, `(subtract (add (multiply 3 4) 7) 5)` can be parsed into the following 
tree

<img src="/part3/semantic-parsing/nla_parse_tree_complex.svg" alt="Parse tree for the expression (subtract (add (multiply 3 4) 7) 5)" width="600"/>

and the linearlized sequence is as follows

```
@start@ -> int
int -> <int,int:int> int int
<int,int:int> -> subtract
int -> <int,int:int> int int
<int,int:int> -> add
int -> <int,int:int> int int
<int,int:int> -> multiply
int -> 3
int -> 4
int -> 7
int -> 5
```

Now how do we incorporate these grammar constraints into our semantic parser? At a high level, we make two changes to the parser we saw in the
previous chapter:

1. Instead of generating one token in the target sequence at each time step in the decoder, we generate one rule in the linearized sequence at one
time step. Since the expression can easily be recovered from the sequence of grammar rules, this will effectively produce the target sequence.

2. We will constrain the decoder to only allow valid non-terminal expansions. We apply these constraints both at training time and at decoding time.
We will go over how we achieve this next.


</exercise>


<exercise id="3" title="Defining a domain-specific (target) language">

To incorporate the grammar into our semantic parser, we need to define the target language as a
[`DomainLanguage`](https://github.com/allenai/allennlp-semparse/blob/master/allennlp_semparse/domain_languages/domain_language.py).
When you define a `DomainLanguage`, all you need to do is specify the constants and functions in your language. The specification includes
the values they should evaluate to and their type-signatures, which for constants are the types of their values, and for the functions are the types of their inputs and outputs.

For example, in our NLA language, `'7'` is a constant, and `'add'` is a function. `'7'` should evaluate to the number `7`, whose type is `integer`, and `'add'` should evaluate to a function that takes two numbers, both `integers`, and returns their sum, also an `integer`.

`DomainLanguage` lets you specify all these in Python. Internally this specification is converted into a context-free grammar like the one
shown earlier in this chapter.

Let us see what the the NLA language looks like as a `DomainLanguage` implementation.

```python
class NlaLanguage(DomainLanguage):
    def __init__(self):
        super().__init__(
            start_types={int},
            allowed_constants={
                "0": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8,
                "9": 9,
            },
        )

    @predicate
    def add(self, num1: int, num2: int) -> int:
        return num1 + num2

    @predicate
    def subtract(self, num1: int, num2: int) -> int:
        return num1 - num2

    @predicate
    def multiply(self, num1: int, num2: int) -> int:
        return num1 * num2

    @predicate
    def divide(self, num1: int, num2: int) -> int:
        return num1 // num2 if num2 != 0 else 0

```

The key pieces of this implementation are `start_types`, `allowed_constants`, and the `predicate` functions. 

 The `DomainLanguage` class defined in `allennlp-semparse` provides several features that we did not need for defining our NLA language. 

</exercise>


<exercise id="4" title="Transition functions">

</exercise>


<exercise id="5" title="State tracking">

</exercise>


<exercise id="6" title="Training">

</exercise>


<exercise id="7" title="Decoding">

</exercise>


<exercise id="8" title="Further reading">

</exercise>
