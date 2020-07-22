# PART 3: PRACTICAL TASKS WITH ALLENNLP

## Document Ranking

This chapter provides an example of how to perform document ranking with AllenNLP. We'll take a look at more advanced features such as different kinds of fields, the `TimeDistributed` module, and making our own metrics. The full code for this guide can be found [here](https://github.com/jacobdanovitch/allenrank).

### 1. An overview on ranking

In its simplest form, document ranking is the task of assigning a score to a query-document pair. This is similar to textual entailment, but instead of our labels being entailment, contradiction, and neutral, our labels instead represent if the document is relevant to the query. This can be presented as either binary classification or regression, where our labels are continuous relevance scores between 0 and 1.

There are several different ranking tasks (ad-hoc, retrieval, clickthrough prediction, etc), as well as several large datasets for it such as [MS-MARCO](https://microsoft.github.io/msmarco/) and [ClueWeb](https://lemurproject.org/clueweb09/). Since these datasets tend to be rather large, we'll use the [MIMICS](https://github.com/microsoft/MIMICS) query clarification dataset as a lightweight alternative. Each row contains a query, a clarifying question, and a list of options presented to the user. This is [used in Bing](https://twitter.com/albondarenko2/status/1225802655504781312/photo/1) to help refine search results.

| <!-- -->                            | <!-- -->                                                  |
|-------------------------------------|-----------------------------------------------------------|
| query                               | headaches                                                 |
| question                            | What do you want to know about this medical condition?    |
| candidate answers (options)         | symptom, treatment, causes, diagnosis, diet               |

Each option will come with its own continuous score; our goal is to predict this score as closely as possible, given the query, the question, and the option itself. Since there are no training/validation/testing datasets, you can use [this script](https://github.com/jacobdanovitch/allenrank/blob/master/scripts/data_split.py) to download and split the dataset:

```shell
$ python scripts/data_split.py https://github.com/microsoft/MIMICS/blob/master/data/MIMICS-ClickExplore.tsv\?raw\=true
```

### 2. Beyond `TextField`s and `LabelField`s

The candidate answers provide a different challenge from what we're used to working with. Usually, our input is a text document, which we process using a `TextField`. Now, we have an input that contains multiple text documents (which can each be multiple words). In this case, we'll turn to the `ListField`. From the [documentation](https://docs.allennlp.org/master/api/data/fields/list_field/): 

> "A `ListField` is a list of other fields. You would use this to represent, e.g., a list of answer options that are themselves `TextFields`."

Sounds perfect for our use case! We can make a couple helper functions for our dataset reader like so:

```python
@DatasetReader.register("mimics")
class MIMICSDatasetReader(DatasetReader):
    
    # ...

    def _make_textfield(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        if self.max_tokens:
            tokens = tokens[:self.max_tokens]
        return TextField(tokens, token_indexers=self.token_indexers)

    def _make_listfield(self, documents: List[str]):
        return ListField([self._make_textfield(d) for d in documents])
```

There are two other matters we have to handle as well. The first is how to handle the query and question. You could make these individual text fields, but here, we're going to concatenate them to eventually be used in a BERT model, where they'll be separated by the special `[SEP]` token.

The second is how to handle labels. While labels have normally been discrete, we now have continuous scores. For this reason, we'll forego the use of a `LabelField`. The main utility of a `LabelField` is to index our labels for easy conversion, but we don't have string labels here that we need indexed. We also have multiple labels. If we wanted to, we could use a `ListField[LabelField]` here, but instead, we're just going to use an [`ArrayField`](https://docs.allennlp.org/master/api/data/fields/array_field/):

> A class representing an array, which could have arbitrary dimensions. A batch of these arrays are padded to the max dimension length in the batch for each dimension.

This is a better fit for our dataset. Here's an overview of our `text_to_instance` method; for the full dataset reader, see the code [here](https://github.com/jacobdanovitch/allenrank/blob/master/allenrank/dataset_readers/mimics_reader.py).

```python

    @overrides
    def text_to_instance(
        self,
        query: str, 
        question: str,
        options: List[str],
        labels: List[float] = None
    ) -> Instance:
        
        token_field = self._make_textfield((query, question))
        options_field = self._make_listfield(options)
        
        fields = { 'tokens': token_field, 'options': options_field }
        if labels:       
            fields['labels'] = ArrayField(np.array(labels), padding_value=-1)
        
        return Instance(fields)
```

### 3. Designing our ranking model

Let's start off by understanding our inputs and outputs. Our model is going to accept `tokens` (the concatenation of the query and question), `options` (the `ListField[TextField]` of answer candidates), and `labels` (the `ArrayField` containing a continuous score for each option). The sizes have been annotated inline.

```python
@Model.register("ranker")
class DocumentRanker(Model):
    
    # ...

    def forward( 
        self, 
        tokens: Dict[str, torch,Tensor], # batch * words
        options: List[Dict[str, torch,Tensor]], # batch * num_options * words
        labels: torch.FloatTensor = None # batch * num_options
    ) -> Dict[str, torch.Tensor]:
```

Now, how do we convert our token IDs to embeddings when they're in `ListField` form? `TextFieldEmbedders` facilitate this by accepting an extra `num_wrapping_dims` keyword argument.

```python
    def forward( 
        self, 
        tokens: Dict[str, torch,Tensor], # batch * words
        options: List[Dict[str, torch,Tensor]], # batch * num_options * words
        labels: torch.FloatTensor = None # batch * num_options
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).long()

        embedded_options = self._text_field_embedder(options, num_wrapping_dims=1)
        options_mask = get_text_field_mask(options).long()
```

This tells our embedder that we have 1 "extra" dimension relative to a normal `TextField`; where normal would be `[Batch, Words]`, we have `[Batch, Options, Words]`. If our options were much longer (e.g., full web pages), we might want to tokenize them into sentences, which would give us `[Batch, Options, Sentences, Words]`. In general, a good rule of thumb for this is `num_wrapping_dims=mask.dim()-2`, since a normal text field is 2 dimensional, and anything beyond that is extra. Under the hood, this is implemented using the `TimeDistributed` module, which we'll take a look at momentarily. 

Next up, we need to compute a relevance score between our tokens and each of the options, which will be used to calculate the loss and metrics. We'll make a `Registerable` for this so we can try out a few different architectures:

```python
class RelevanceMatcher(Registrable, nn.Module):
    def __init__(
        self,
        input_dim: int
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, 1, bias=False)


    def forward(
        self, 
        token_embeddings: torch.Tensor, 
        option_embeddings: torch.Tensor,
        token_mask: torch.Tensor = None,
        option_mask: torch.Tensor = None
    ):
        raise NotImplementedError()
```

This outlines what each architecture should ultimately boil down to: it accepts the `tokens` and one `option`, and computes a relevance score. Several others are included in the full codebase, but we'll stick with a very simple one:

```python
@RelevanceMatcher.register('bert_cls')
class BertCLS(RelevanceMatcher):
    def __init__(
        self,
        seq2vec_encoder: Seq2VecEncoder, # should be `cls_pooler`
        **kwargs
    ):
        kwargs['input_dim'] = seq2vec_encoder.get_output_dim()*4
        super().__init__(**kwargs)

        self._seq2vec_encoder = seq2vec_encoder

    def forward(
        self, 
        token_embeddings: torch.Tensor, 
        option_embeddings: torch.Tensor,
        token_mask: torch.Tensor = None,
        option_mask: torch.Tensor = None
    ):

        tokens_encoded = self._seq2vec_encoder(token_embeddings, query_mask)
        option_encoded = self._seq2vec_encoder(option_embeddings, option_mask)

        interaction_vector = torch.cat([tokens_encoded, option_encoded, token_encoded-option_encoded, tokens_encoded*option_encoded], dim=1)
        dense_out = self.dense(interaction_vector)
        score = torch.squeeze(dense_out,1)

        return score
```

This will simply take the `[CLS]` token from both `tokens` and `option`, compute an interaction vector (a la [InferSent](https://research.fb.com/publications/supervised-learning-of-universal-sentence-representations-from-natural-language-inference-data/)), and predict a dense score. There are much fancier alternatives, but this will do for now.

### 4. The `TimeDistributed` module

Let's revisit the `forward` method of our `DocumentRanker`.

```python
    def forward( 
        self, 
        tokens: Dict[str, torch,Tensor], # batch * words
        options: List[Dict[str, torch,Tensor]], # batch * num_options * words
        labels: torch.FloatTensor = None # batch * num_options
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).long()

        embedded_options = self._text_field_embedder(options, num_wrapping_dims=1)
        options_mask = get_text_field_mask(options).long()
```

We have a module to score each query/document pair, but our data is formatted as `(tokens, options)`. We need to align the query to each option for the relevance layer to work. 

First, we'll use `torch.expand` to create a copy of the token embeddings for each option. Since `torch.expand` doesn't use additional memory, this doesn't harm space complexity, but time complexity will still increase. 

```python
# [batch, num_options, words, dim]
embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_options.size(1), -1, -1)
mask = mask.unsqueeze(1).expand(-1, embedded_options.size(1), -1)
```

Now, we have an identical copy of the token embeddings for each option embedding. From here, it's easier to stop worrying about the number of options and simply flatten everything to `[batch, words, dim]`. Luckily, there's a module to handle this for us; the `TimeDistributed` module will flatten each input tensor for us. We simply wrap our relevance matcher like so:

```python
self.relevance_matcher = TimeDistributed(self.relevance_matcher)
```

Now we can use our relevance matcher to score each pair.

```python
    def forward( 
        self, 
        tokens: Dict[str, torch,Tensor], # batch * words
        options: List[Dict[str, torch,Tensor]], # batch * num_options * words
        labels: torch.FloatTensor = None # batch * num_options
    ) -> Dict[str, torch.Tensor]:
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens).long()

        embedded_options = self._text_field_embedder(options, num_wrapping_dims=1)
        options_mask = get_text_field_mask(options).long()

        embedded_text = embedded_text.unsqueeze(1).expand(-1, embedded_options.size(1), -1, -1)
mask = mask.unsqueeze(1).expand(-1, embedded_options.size(1), -1)


        scores = self._relevance_matcher(embedded_text, embedded_options, mask, options_mask).squeeze(-1)
        probs = torch.sigmoid(scores)
```

### 5. Evaluating with custom metrics