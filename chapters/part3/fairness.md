---
title: 'Fairness and Bias Mitigation'
description: A practical guide into the AllenNLP Fairness module.
author: Arjun Subramonian 
type: chapter
---

<textblock>
As models and datasets become increasingly large and complex, it is critical to evaluate the fairness of models according to multiple definitions of fairness and mitigate biases in learned representations. <code>allennlp.fairness</code> aims to make fairness metrics, fairness training tools, and bias mitigation algorithms extremely easy to use and accessible to researchers and practitioners of all levels.
<br>
<br>
We hope <code>allennlp.fairness</code> empowers everyone in NLP to combat algorithmic bias, that is, the "unjust, unfair, or prejudicial treatment of people related to race, income, sexual orientation, religion, gender, and other characteristics historically associated with discrimination and marginalization, when and where they manifest in algorithmic systems or algorithmically aided decision-making" (Chang et al. 2019). Ultimately, "people who are the most marginalized, people who’d benefit the most from such technology, are also the ones who are more likely to be systematically excluded from this technology" because of algorithmic bias (Chang et al. 2019). 
</textblock>

<exercise id="1" title="Why do we need fairness and bias mitigation tools?">

This section provides a high-level overview of fairness and how NLP models can <em>encode</em> and <em>amplify</em> biases.  Readers are encouraged to engage with original research on fairness and biases; we recommend the following papers:

1. Blodgett, S.L., Barocas, S., Daumé, H., & Wallach, H. (2020). Language (Technology) is Power: [A Critical Survey of "Bias" in NLP](https://api.semanticscholar.org/CorpusID:218971825). ACL.

2. Chang, K.W., Ordonez, V., Mitchell, M., Prabhakaran, V (2019). [Tutorial: Bias and Fairness in Natural Language Processing](http://web.cs.ucla.edu/~kwchang/talks/emnlp19-fairnlp/). EMNLP 2019.

3. Barocas, S., Hardt, M., and Narayanan, A. 2019. [Fairness and machine learning](https://fairmlbook.org).

4. Bender, E.M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://api.semanticscholar.org/CorpusID:232040593). Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.

In the rest of this section, we motivate the need for fairness and bias mitigation tools in NLP, based on References 1, 2 and 3 above.

<em>Fairness</em> concerns ensuring that a model performs equitably for all groups of people with respect to protected attributes. For instance, we may want a model to correctly predict `[MASK]` tokens at the same rate for men, women, and non-binary genders.

<em>Biases</em> are stereotypical/unjust associations a model encodes with respect to protected attributes, or subpar model performance for certain groups. Hence, <em>bias mitigation</em> is the process of lessening the severity of stereotypical/unjust associations and disparate model performance. For example, we may want to disassociate all occupations from gender, to mitigate gender bias in a downstream application that recommends jobs to individuals. 

## Biases in Data

We are all aware that large language models are hungry for text data, and these data greatly influence the word and sentence-level representations the models learn. NLP researchers and practitioners often scrape data from the Internet, which include news articles, Wikipedia pages, and even patents! But, how representative are the text in these data of the real-world frequencies of words among different groups of people? The answer is **not at all**.
<br>
<br>
This is the result of <em>reporting bias</em>, in which the "frequency with which people write about actions, outcomes, or properties is not a reflection of real-world frequencies or the degree to which a property is characteristic of a class of individuals" [2]. For instance, "murdered" occurs with a much higher frequency than does "blinked" in text corpuses because journalists usually report about murders rather than blinking, as a result of the relative rarity of murders [2].

There are many other biases in data that affect language models [2]:
1. **Selection bias:** Data selection does not reflect a random sample, e.g. English language data do not represent every dialect of English, Wikipedia pages overrepresent men

2. **Out-group homogeneity bias:** People tend to see outgroup members as more alike than ingroup members when comparing attitudes, values, personality traits, and other characteristics, e.g. [researchers incorrectly treat "non-binary" as a single, cohesive gender](https://venturebeat.com/2020/07/14/study-describes-facial-recognition-system-designed-to-identify-non-binary-people/)

3. **Biased data representation:** Even with an appropriate amount of data for every group, some groups are represented less positively than others, e.g. [Wikipedia pages exhibit serious biases against women](https://wikimediafoundation.org/news/2018/10/18/wikipedia-mirror-world-gender-biases/)

4. **Biased labels:** Annotations in dataset reflect the worldviews of annotators, e.g. [annotators do not add wedding-related labels to images of wedding traditions from different parts of the world, due to an unawareness of non-Western cultures and traditions](https://ai.googleblog.com/2018/09/introducing-inclusive-images-competition.html)

Barocas and Selbst, in their 2016 paper "[Big Data's Disparate Impact](https://api.semanticscholar.org/CorpusID:143133374)," also broke down biases in data into the following categories (which overlap with the categories above) [3]:

1. **Skewed sampling (feedback loop):** Future observations confirm a model's predictions, which leads to fewer opportunities to make observations that contradict the model's predictions, thereby causing any biases to compound, e.g. predictive policing tools are influenced by their own predictions of where crime will occur

2. **Tainted examples:** Social biases like racism, sexism, and homophobia taint datasets, and then get baked into models which are trained and evaluated on these datasets, e.g. Amazon's automated resume screening system identified women to be unfit for engineering roles because of the company's poor track record of hiring women

3. **Limited features:** Features may be less informative or less reliably collected for certain groups of people, which leads to a model exhibiting disparate errors for different groups

4. **Sample size disparities:** There exist disparities in the amount of data across certain groups of people, which leads to a model exhibiting disparate errors for different groups

5. **Proxies:** Even after the removal of a feature which a model should not consider (e.g. race), there exist other features with a high correlation with the undesirable feature that the model can (unfairly) leverage (e.g. zip code)

Please note that every dataset is biased, and it is <em>impossible</em> to fully remove biases from datasets; even though it may be relatively easy to identify and remove explicit social biases, it is difficult to eliminate skewed samples and proxies that could reinforce these biases. That being said, it's important to be cognizant of and transparent about the biases your datasets could have as a first step towards controlling them. Hence, it's a great idea to create [datasheets for your datasets](https://api.semanticscholar.org/CorpusID:4421027).

## Biases in Models

You may often hear, in the context of biases, "It's not the model, it's the data!" In fact, models are excellent at <em>encoding</em> and <em>amplifying</em> biases.

1. **Imbalanced datasets:** Model performance inherently favors groups with rich features and a large, diverse sample, thereby marginalizing groups with limited features and a poor sample size.

2. **Spurious correlations:** Models learn complex, non-linear and spurious correlations between features and annotations that often reflect social biases and are difficult to detect and eliminate. For example, [Wang et al.](https://api.semanticscholar.org/CorpusID:195847929) discovered that even after removing subjects from images in a binary gender-balanced dataset, a model could still easily predict the gender of subjects because women were photographed overwhelmingly frequently in the kitchen, that is, the model leveraged the existence of a kitchen in the image as a proxy for gender. Clearly, balanced datasets alone are not effective at taming bias amplification.

3. **Contextualization:** LSTMs and attention-based models, through contextualization, allow for biases to be propagated across words in a sentence.

## Biases in Interpretation

Beyond biases in data and models, there also exist biases in interpretation [2]:

1. **Confirmation bias:** The tendency to search for, interpret, favor, recall information in a way that confirms preexisting beliefs, e.g. a company misconstrues its automated resume screening system's identification of women as unfit for engineering roles (because of the company's poor track record of hiring women) as evidence that women are unfit for engineering

2. **Overgeneralization:** Coming to conclusions based on information that is too general, e.g. researchers conclude that a facial recognition model works well for everyone based on its overall accuracy on data containing mostly white people

3. **Correlation fallacy:** Confusing correlation with causation

4. **Automation bias:** Propensity for humans to favor suggestions from automated decision-making systems over contradictory information without automation

Biases in interpretation truly necessitate the creation and usage of metrics that can uncover unfairness and biased associations in models. 

## Call to Action

Why should we care about fairness and biases? Language models are increasingly used throughout everyday life, and as they become more prevalent in critical decision-making systems which impact people, their unfair encoding and amplification of biases can pose serious harms for already-marginalized communities. In NLP literature, harms are often categorized into the following (Barocas and Selbst, 2016):

1. **Representational harms:** when systems reinforce the subordination of some groups along the lines of identity
2. **Allocational harms:** when a system allocates or withholds a certain opportunity or resource

It is important to ensure fairness and mitigate biases even for seemingly trivial NLP tasks, such as named entity recognition, as these are often used as building blocks for more complex downstream applications like text summarization and even identity verification. This is one of many reasons to use [model cards for model reporting](https://dl.acm.org/doi/10.1145/3287560.3287596).

Can biases be removed? No, [completely removing biases is difficult](https://api.semanticscholar.org/CorpusID:73729169), especially with large language models, because of contextualization, spurious correlations, and latent variables. However, biases can be reduced and controlled. There doesn't exist a universal point at which to stop mitigating biases; bias mitigation must go hand-in-hand with the auditing of real-world model performance to ensure that humans are not unfairly impacted by the model's decisions.

</exercise>

<exercise id="2" title="An Overview of Fairness Metrics">

`allennlp.fairness.fairness_metrics` offers three different fairness metrics, all of which represent mutually-exclusive notions of fairness, except in degenerate cases.

Let `C` represent our `predicted_labels`, `A` represent our `protected_variable_labels`, and `Y` represent our `gold_labels`.

1. **[Independence](http://docs.allennlp.org/main/api/fairness/fairness_metrics/#independence)** measures the statistical independence of a protected variable from predictions. It has been explored through many equivalent terms or variants, such as demographic parity, statistical parity, group fairness, and disparate impact. Informally, `Independence` measures how much `P(C | A)` diverges from `P(C)`.

2. **[Separation](http://docs.allennlp.org/main/api/fairness/fairness_metrics/#separation)** allows correlation between the predictions and a protected variable to the extent that it is justified by the gold labels. Informally, `Separation` measures how much `P(C | A, Y)` diverges from `P(C | Y)`.

3. **[Sufficiency](http://docs.allennlp.org/main/api/fairness/fairness_metrics/#sufficiency)** is satisfied by the predictions when the protected variable and gold labels are clear from context. Informally, `Sufficiency` measures how much `P(Y | A, C)` diverges from `P(Y | C)`.

All these fairness metrics can be used like any other `Metric` in the library, and they are supported in distributed settings. Here is an example code snippet for using these metrics with your own models:

```python
from allennlp.models.model import Model
from allennlp.fairness.fairness_metrics import Independence, Separation, Sufficiency

class YourModel(Model):
    
    def __init__(self, *args, **kwargs):
        ...
        # Initialize fairness metric objects
        self._independence = Independence()
        self._separation = Separation()
        self._sufficiency = Sufficiency()
        ...

    def forward(self, *args, **kwargs):
        ...
        # Accumulate metrics over batches
        self._independence(predicted_labels, protected_variable_labels)
        self._separation(predicted_labels, gold_labels, protected_variable_labels)
        self._sufficiency(predicted_labels, gold_labels, protected_variable_labels)
        ...

model = YourModel(...)
...
# Get final values of metrics after all batches have been processed
print(model._independence.get_metric(), model._separation.get_metric(), model._sufficiency.get_metric())
```

Please refer to the [documentation](http://docs.allennlp.org/main/api/fairness/fairness_metrics/) for `allennlp.fairness.fairness_metrics` for more details.

</exercise>

<exercise id="3" title="An Overview of Bias Mitigation and Bias Direction Methods">

## Bias Mitigators

`allennlp.fairness.bias_mitigators` offers a suite of differentiable methods to mitigate biases for binary concepts in static embeddings. These methods can be used functionally (please refer the [documentation](http://docs.allennlp.org/main/api/fairness/bias_mitigators/) for further details, as this section explains them conceptually), but [Section 5](/fairness#5) will explain how to use them in practice, with large, contextual language models.

**Note:** In the examples below, we treat gender identity as binary, which does not accurately characterize gender in real life. The examples and graphics are taken from Rathore, A., Dev, S., Phillips, J.M., Srikumar, V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021). [VERB: Visualizing and Interpreting Bias Mitigation Techniques for Word Representations](https://api.semanticscholar.org/CorpusID:233168618). ArXiv, abs/2104.02797.

1. **[HardBiasMitigator](http://docs.allennlp.org/main/api/fairness/bias_mitigators/#hardbiasmitigator)** mitigates biases in embeddings by:

• <em>Neutralizing:</em> ensuring protected variable-neutral words (e.g. `receptionist`, `banker`, `engineer`, `nurse`) remain equidistant from the bias direction (`gender`) by removing component of embeddings in the bias direction.
<br>
• <em>Equalizing:</em> ensuring that protected variable-related word pairs (e.g. `(brother, sister)`, `(woman, man)`) are averaged out to have the same norm and are equidistant from the bias direction.

![hard_bias_mitigator](/part3/fairness/hard-debiasing.png) 

2. **[LinearBiasMitigator](http://docs.allennlp.org/main/api/fairness/bias_mitigators/#linearbiasmitigator)** mitigates biases in all embeddings (e.g. `receptionist`, `banker`, `woman`, `man`) by simply removing their component in the bias direction (`gender`).

![linear_bias_mitigator](/part3/fairness/linear-debiasing.png) 

3. **[INLPBiasMitigator](http://docs.allennlp.org/main/api/fairness/bias_mitigators/#inlpbiasmitigator)** mitigates biases by repeatedly building a linear classifier that separates concept groups (e.g. (e.g. `engineer`, `homemaker`, `woman`, `man`) and linearly projecting all words along the classifier normal.

![inlp_bias_mitigator](/part3/fairness/inlp-debiasing.png) 

4. **[OSCaRBiasMitigator](http://docs.allennlp.org/main/api/fairness/bias_mitigators/#oscarbiasmitigator)** mitigates biases in embeddings by disassociating concept subspaces through subspace orthogonalization. Informally, OSCaR applies a graded rotation on the embedding space to rectify two ideally-independent concept subspaces (e.g. `gender` and `occupation`) so that they become orthogonal.

![oscar_bias_mitigator](/part3/fairness/oscar-debiasing.png)

## Bias Direction

Many of the above bias mitigators require a <em>predetermined bias direction</em>. Hence, `allennlp.fairness.bias_direction` offers a suite of differentiable methods to compute the bias direction or concept subspace representing binary protected variables from seed words.

**Note:** In the examples below, we treat gender identity as binary, which does not accurately characterize gender in real life. The examples and graphics are taken from Rathore, A., Dev, S., Phillips, J.M., Srikumar, V., Zheng, Y., Yeh, C.M., Wang, J., Zhang, W., & Wang, B. (2021). [VERB: Visualizing and Interpreting Bias Mitigation Techniques for Word Representations](https://api.semanticscholar.org/CorpusID:233168618). ArXiv, abs/2104.02797.

1. **[PCABiasDirection](http://docs.allennlp.org/main/api/fairness/bias_direction/#pcabiasdirection)** computes a one-dimensional subspace that is the span of a specific concept (e.g. `gender`) using PCA. This subspace minimizes the sum of squared distances from all seed word embeddings (e.g. `man`, `woman`, `brother`, `sister`).

![pca](/part3/fairness/pca.png)

2. **[PairedPCABiasDirection](http://docs.allennlp.org/main/api/fairness/bias_direction/#pairedpcabiasdirection)** computes a one-dimensional subspace that is the span of a specific concept (e.g. `gender`) as the first principle component of the difference vectors between seed word embedding pairs (e.g. `(man, woman)`), (`(brother, sister)`).

![paired_pca](/part3/fairness/paired-pca.png)

3. **[TwoMeansBiasDirection](http://docs.allennlp.org/main/api/fairness/bias_direction/#twomeansbiasdirection)** computes a one-dimensional subspace that is the span of a specific concept (e.g. `gender`) as the normalized difference vector of the averages of seed word embedding sets (e.g. `[sister, woman, she]`, `[brother, man, he]`).

![two_means](/part3/fairness/two-means.png)

4. **[ClassificationNormalBiasDirection](http://docs.allennlp.org/main/api/fairness/bias_direction/#classificationnormalbiasdirection)** computes a one-dimensional subspace that is the span of a specific concept (e.g. `gender`) as the direction perpendicular to the classification boundary of a linear support vector machine fit to classify seed word embedding sets (e.g. `[sister, woman, she]`, `[brother, man, he]`).

![classification_normal](/part3/fairness/classification-normal.png)

</exercise>

<exercise id="4" title="An Overview of Bias Metrics">

`allennlp.fairness.bias_metrics` offers a suite of metrics to quantify how much bias is encoded by word embeddings and determine the effectiveness of bias mitigation. These methods can be used functionally (please refer the [documentation](http://docs.allennlp.org/main/api/fairness/bias_metrics/) for further details, as this section explains them conceptually), but [Section 5](/fairness#5) will explain how to use them in practice, with large, contextual language models.

**Note:** In the examples below, we treat gender identity as binary, which does not accurately characterize gender in real life. 

1. **[WordEmbeddingAssociationTest](http://docs.allennlp.org/main/api/fairness/bias_metrics/#wordembeddingassociationtest)** measures the likelihood there is no difference between two sets of target words in terms of their relative similarity to two sets of attribute words by computing the probability that a random permutation of attribute words would produce the observed (or greater) difference in sample means. Typical values are around [-1, 1], with values closer to 0 indicating less biased associations. It is the analog of the Implicit Association Test from psychology for <em>static</em> word embeddings.

```python
import torch
...
target_embeddings1 = torch.cat([engineer, doctor])
target_embeddings2 = torch.cat([receptionist, nurse])
attribute_embeddings1 = torch.cat([man, brother, boy])
attribute_embeddings2 = torch.cat([woman, sister, girl])
# Print out WEAT score
print(WordEmbeddingAssociationTest()(target_embeddings1, target_embeddings2, attribute_embeddings1, attribute_embeddings2))
```

2. **[EmbeddingCoherenceTest](http://docs.allennlp.org/main/api/fairness/bias_metrics/#embeddingcoherencetest)** measures if groups of words have stereotypical associations by computing the Spearman Coefficient of lists of attribute embeddings sorted based on their similarity to target embeddings. Score ranges from [-1, 1], with values closer to 1 indicating less biased associations.

```python
import torch
...
target_embeddings = torch.cat([engineer, doctor, receptionist, nurse])
attribute_embeddings1 = torch.cat([man, brother, boy])
attribute_embeddings2 = torch.cat([woman, sister, girl])
# Print out ECT score
print(EmbeddingCoherenceTest()(target_embeddings, attribute_embeddings1, attribute_embeddings2))
```

3. **[NaturalLanguageInference](http://docs.allennlp.org/main/api/fairness/bias_metrics/#naturallanguageinference)** measures the effect biased associations have on textual entailment decisions made downstream, given neutrally-constructed pairs of sentences differing only in the subject. Below are examples of neutrally-constructed sentence pairs:

```python
{"annotator_labels": ["neutral"], "gold_label": "neutral", "sentence1": "An accountant bought a bus.", "sentence2": "A gentleman bought a bus."}
{"annotator_labels": ["neutral"], "gold_label": "neutral", "sentence1": "An accountant bought a car.", "sentence2": "A lady bought a car."}
{"annotator_labels": ["neutral"], "gold_label": "neutral", "sentence1": "A doctor bought a heater.", "sentence2": "A man bought a heater."}
{"annotator_labels": ["neutral"], "gold_label": "neutral", "sentence1": "A doctor bought a cat.", "sentence2": "A woman bought a cat."}
```
`NaturalLanguageInference` computes three different submetrics:

• **Net Neutral (NN):** The average probability of the neutral label being predicted by an SNLI model across all sentence pairs.

• **Fraction Neutral (FN):** The fraction of sentence pairs predicted neutral by the SNLI model.

• **Threshold:tau (T:tau):** A parameterized measure that reports the fraction of examples whose probability of neutral is above tau.

For all submetrics, a value closer to 1 suggests lower bias, as bias will result in a higher probability of entailment or contradiction.

`NaturalLanguageInference` can be used like any other `Metric` in the library, and it is supported in distributed settings. Here is an example code snippet for using this metric with your own models:

```python
from allennlp.models.model import Model
from allennlp.fairness.bias_metrics import NaturalLanguageInference

class YourModel(Model):
    
    def __init__(self, *args, **kwargs):
        ...
        # Initialize NaturalLanguageInference metric object
        self._nli = NaturalLanguageInference()
        ...

    def forward(self, *args, **kwargs):
        ...
        # Accumulate metric over batches
        self._nli(nli_probabilities)
        ...

model = YourModel(...)
...
# Get final values of metric after all batches have been processed
print(model._nli.get_metric())
```

4. **[AssociationWithoutGroundTruth](http://docs.allennlp.org/main/api/fairness/bias_metrics/#associationwithoutgroundtruth)** measures model biases in the absence of ground truth. It does so by computing one-vs-all or pairwise association gaps using statistical measures like `nPMIxy`, `nPMIy`, `PMI^2`, and `PMI`, which are capable of capturing labels across a range of marginal frequencies. A gap of nearly 0 implies less bias on the basis of Association in the Absence of Ground Truth.

`AssociationWithoutGroundTruth` can be used like any other `Metric` in the library, and it is supported in distributed settings. Here is an example code snippet for using this metric with your own models:

```python
from allennlp.models.model import Model
from allennlp.fairness.bias_metrics import AssociationWithoutGroundTruth

class YourModel(Model):
    
    def __init__(self, *args, **kwargs):
        ...
        # Initialize AssociationWithoutGroundTruth metric object
        self._npmixy = AssociationWithoutGroundTruth()
        ...

    def forward(self, *args, **kwargs):
        ...
        # Accumulate metric over batches
        self._npmixy(predicted_labels, protected_variable_labels)
        ...

model = YourModel(...)
...
# Get final values of metric after all batches have been processed
print(model._npmixy.get_metric())
```
</exercise>

<exercise id="5" title="Applying Bias Mitigation to Large, Contextual Language Models">

The previous sections detail bias mitigation and direction methods and bias metrics. However, you might be wondering, how can I apply these tools directly from a config file? Additionally, you probably noticed that they are intended for <em>static</em> word embeddings, not the large, contextual language models you're likely interested in using.

Fortunately, `allennlp.fairness` provides a `Model` wrapper called [`BiasMitigatorApplicator`](http://docs.allennlp.org/main/api/fairness/bias_direction_wrappers/) to mitigate biases in contextual embeddings during finetuning on a downstream task and test time, based on Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [On Measuring and Mitigating Biased Inferences of Word Embeddings](https://api.semanticscholar.org/CorpusID:201670701). Proceedings of the AAAI Conference on Artificial Intelligence, 34(05), 7659-7666. https://doi.org/10.1609/aaai.v34i05.6267. In this section, we detail how to apply `BiasMitigatorApplicator` to reduce binary gender bias in a large RoBERTA model for SNLI.

1. The first step is to obtain the pretained model for which you would like to reduce binary gender bias. In this case, we will use [AllenNLP's public large RoBERTA model for SNLI](https://storage.googleapis.com/allennlp-public-models/snli-roberta.2021-03-11.tar.gz).

2. `BiasMitigatorApplicator` works by applying a bias mitigator of your choice directly after the static embedding layer of the pretrained RoBERTA model during finetuning and evaluation. This reduces the model's ability to propagate gender biases across words in input sentences through contextualization. In a simplistic sense, the mitigator implicitly functions as a linear adversary that forces the model to discover alternate latent variables or features besides gender that it can use to perform well on SNLI.

`BiasMitigatorApplicator` works with any Transformer model that has a static embedding layer. However, the relationships between (1) the number of epochs of finetuning, (2) the number of layers in the model, (3) the type of bias mitigator and bias direction used, and the efficacy of bias mitigation is not well-studied and a promising direction for future research. While finetuning for more epochs with the bias mitigator might push the model to depend less on gender, it could also cause the model to overfit to spurious correlations and/or discover proxies for gender that are not contained within the gender bias subspace of the mitigator. Furthermore, it is likely harder to mitigate biases for models with a larger number of layers, as they are capable of learning extremely complex mappings. And, how well a particular type of bias mitigator and bias direction work strongly depends on the finetuning data and model architecture. You should not assume that using a `BiasMitigator` resolves any particular bias issue in a model. There is [published literature](https://api.semanticscholar.org/CorpusID:201670701) suggesting that these techniques should help (without noticeably compromising model performance on the test set), but that needs to be evaluated for any actual downstream use.

Below is an [example config file](https://github.com/allenai/allennlp-models/blob/main/training_config/pair_classification/binary_gender_bias_mitigated_snli_roberta.jsonnet) for finetuning a pretrained RoBERTA model for SNLI using `BiasMitigatorApplicator` with linear bias mitigation and a two-means bias direction. As mentioned previously, `BiasMitigatorApplicator` simply wraps the pretrained model. `two_means` requires a `seed_word_pairs_file` and tokenizer to tokenize the words in said file. `seed_word_pairs_file` must follow the format in the example: `[["woman", "man"], ["girl", "boy"], ["she", "he"], ...]`. And that's it!

```python
local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader":{
    "type": "snli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
  "model": {
    "type": "allennlp.fairness.bias_mitigator_applicator.BiasMitigatorApplicator", 
    "base_model": {
      "_pretrained": {
        "archive_file": "https://storage.googleapis.com/allennlp-public-models/snli-roberta.2021-03-11.tar.gz",
        "module_path": "",
        "freeze": false
      }
    },
    "bias_mitigator": {
      "type": "linear",
      "bias_direction": {
        "type": "two_means",
        "seed_word_pairs_file": "https://raw.githubusercontent.com/tolga-b/debiaswe/4c3fa843ffff45115c43fe112d4283c91d225c09/data/definitional_pairs.json",
        "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
      }
    }
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}
```

You can run this config file with the following CLI command: `allennlp train config.jsonnet --include-package allennlp_models -s /tmp/snli`.

3. After just two epochs of finetuning (we chose model with highest accuracy on validation set), we can observe a considerable reduction in binary gender bias with respect to occupations, as measured by the `NaturalLanguageInference` bias metric on a [large dataset of neutrally-constructed textual entailment sentence pairs](https://storage.googleapis.com/allennlp-public-models/binary-gender-bias-mitigated-snli-dataset.jsonl). Even better, this comes without a noticeable difference in performance on the test set!

To thoroughly evaluate the reduction in binary gender bias with respect to occupations, we will write an `EvaluateBiasMitigation` subcommand:

```python
"""
The `evaluate-bias-mitigation` subcommand can be used to
compare a bias-mitigated, finetuned model with a baseline, pretrained model
against an SNLI dataset following the format in [On Measuring
and Mitigating Biased Inferences of Word Embeddings]
(https://arxiv.org/pdf/1908.09369.pdf) and reports the
Net Neutral, Fraction Neutral, and Threshold:tau metrics.
"""

import argparse
import json
import logging
from typing import Any, Dict, Tuple
from overrides import overrides
import tempfile
import torch
import os

from allennlp.commands.subcommand import Subcommand
from allennlp.common import logging as common_logging
from allennlp.common.util import prepare_environment
from allennlp.fairness.bias_metrics import NaturalLanguageInference
from allennlp.data import DataLoader
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate

logger = logging.getLogger(__name__)


@Subcommand.register("evaluate-bias-mitigation")
class EvaluateBiasMitigation(Subcommand):
    @overrides
    def add_subparser(self, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
        description = """Evaluate bias mitigation"""
        subparser = parser.add_parser(
            self.name, description=description, help="Evaluate bias mitigation."
        )

        subparser.add_argument(
            "bias_mitigated_archive_file",
            type=str,
            help="path to a bias-mitigated archived trained model",
        )

        subparser.add_argument(
            "baseline_archive_file", type=str, help="path to a baseline archived trained model"
        )

        subparser.add_argument(
            "input_file", type=str, help="path to the file containing the SNLI evaluation data"
        )

        subparser.add_argument("batch_size", type=int, help="batch size to use during evaluation")

        subparser.add_argument(
            "--bias-mitigated-output-file",
            type=str,
            help="optional path to write the metrics to as JSON",
        )

        subparser.add_argument(
            "--baseline-output-file",
            type=str,
            help="optional path to write the metrics to as JSON",
        )

        subparser.add_argument(
            "--bias-mitigated-predictions-file",
            type=str,
            help="optional path to write bias-mitigated predictions to as JSON",
        )

        subparser.add_argument(
            "--baseline-predictions-file",
            type=str,
            help="optional path to write baseline predictions to as JSON",
        )

        subparser.add_argument(
            "--predictions-diff-output-file",
            type=str,
            help="optional path to write diff of bias-mitigated and baseline predictions to as JSON",
        )

        subparser.add_argument(
            "--taus",
            type=float,
            nargs="+",
            default=[0.5, 0.7],
            help="tau parameters for Threshold metric",
        )

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(
            "--cuda-device", type=int, default=-1, help="id of GPU to use (if any)"
        )

        subparser.add_argument(
            "--bias-mitigation-overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the bias mitigation experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--baseline-overrides",
            type=str,
            default="",
            help=(
                "a json(net) structure used to override the baseline experiment configuration, e.g., "
                "'{\"iterator.batch_size\": 16}'.  Nested parameters can be specified either"
                " with nested dictionaries or with dot syntax."
            ),
        )

        subparser.add_argument(
            "--file-friendly-logging",
            action="store_true",
            default=False,
            help="outputs tqdm status on separate lines and slows tqdm refresh rate",
        )

        subparser.set_defaults(func=evaluate_from_args)

        return subparser


class SNLIPredictionsDiff:
    def __init__(self):
        self.diff = []

    def __call__(self, bias_mitigated_labels, baseline_labels, original_tokens, tokenizer):
        """
        Returns label changes induced by bias mitigation and the corresponding sentence pairs.
        """
        for idx, label in enumerate(bias_mitigated_labels):
            if label != baseline_labels[idx]:
                self.diff.append(
                    {
                        "sentence_pair": tokenizer.convert_tokens_to_string(original_tokens[idx]),
                        "bias_mitigated_label": label,
                        "baseline_label": baseline_labels[idx],
                    }
                )

    def get_diff(self):
        return self.diff


# TODO: allow bias mitigation and baseline evaluations to run simultaneously on
# two different GPUs
def evaluate_from_args(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    common_logging.FILE_FRIENDLY_LOGGING = args.file_friendly_logging

    # Disable some of the more verbose logging statements
    logging.getLogger("allennlp.common.params").disabled = True
    logging.getLogger("allennlp.nn.initializers").disabled = True
    logging.getLogger("allennlp.modules.token_embedders.embedding").setLevel(logging.INFO)

    # Load from bias-mitigated archive
    bias_mitigated_archive = load_archive(
        args.bias_mitigated_archive_file,
        cuda_device=args.cuda_device,
        overrides=args.bias_mitigation_overrides,
    )
    bias_mitigated_config = bias_mitigated_archive.config
    prepare_environment(bias_mitigated_config)
    bias_mitigated_model = bias_mitigated_archive.model
    bias_mitigated_model.eval()

    # Load from baseline archive
    baseline_archive = load_archive(
        args.baseline_archive_file, cuda_device=args.cuda_device, overrides=args.baseline_overrides
    )
    baseline_config = baseline_archive.config
    prepare_environment(baseline_config)
    baseline_model = baseline_archive.model
    baseline_model.eval()

    # Load the evaluation data
    bias_mitigated_dataset_reader = bias_mitigated_archive.validation_dataset_reader
    baseline_dataset_reader = baseline_archive.validation_dataset_reader

    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)

    bias_mitigated_data_loader_params = bias_mitigated_config.pop("validation_data_loader", None)
    if bias_mitigated_data_loader_params is None:
        bias_mitigated_data_loader_params = bias_mitigated_config.pop("data_loader")
    # override batch sampler if exists
    if "batch_sampler" in bias_mitigated_data_loader_params:
        del bias_mitigated_data_loader_params["batch_sampler"]
    bias_mitigated_data_loader_params["batch_size"] = args.batch_size
    bias_mitigated_data_loader = DataLoader.from_params(
        params=bias_mitigated_data_loader_params,
        reader=bias_mitigated_dataset_reader,
        data_path=evaluation_data_path,
    )
    bias_mitigated_data_loader.index_with(bias_mitigated_model.vocab)

    baseline_data_loader_params = baseline_config.pop("validation_data_loader", None)
    if baseline_data_loader_params is None:
        baseline_data_loader_params = baseline_config.pop("data_loader")
    # override batch sampler if exists
    if "batch_sampler" in baseline_data_loader_params:
        del baseline_data_loader_params["batch_sampler"]
    baseline_data_loader_params["batch_size"] = args.batch_size
    baseline_data_loader = DataLoader.from_params(
        params=baseline_data_loader_params,
        reader=baseline_dataset_reader,
        data_path=evaluation_data_path,
    )
    baseline_data_loader.index_with(baseline_model.vocab)

    if args.bias_mitigated_predictions_file:
        bias_mitigated_filename = args.bias_mitigated_predictions_file
        bias_mitigated_file = os.open(bias_mitigated_filename, os.O_RDWR)
    else:
        bias_mitigated_file, bias_mitigated_filename = tempfile.mkstemp()
    bias_mitigated_output_metrics = evaluate(
        bias_mitigated_model,
        bias_mitigated_data_loader,
        args.cuda_device,
        predictions_output_file=bias_mitigated_filename,
    )

    if args.baseline_predictions_file:
        baseline_filename = args.baseline_predictions_file
        baseline_file = os.open(baseline_filename, os.O_RDWR)
    else:
        baseline_file, baseline_filename = tempfile.mkstemp()
    baseline_output_metrics = evaluate(
        baseline_model,
        baseline_data_loader,
        args.cuda_device,
        predictions_output_file=baseline_filename,
    )

    create_diff = hasattr(baseline_dataset_reader, "_tokenizer")
    if create_diff:
        diff_tool = SNLIPredictionsDiff()
    bias_mitigated_nli = NaturalLanguageInference(
        neutral_label=bias_mitigated_model.vocab.get_token_index("neutral", "labels"),
        taus=args.taus,
    )
    baseline_nli = NaturalLanguageInference(
        neutral_label=baseline_model.vocab.get_token_index("neutral", "labels"),
        taus=args.taus,
    )
    with open(bias_mitigated_file, "r") as bias_mitigated_fd, open(
        baseline_file, "r"
    ) as baseline_fd:
        for bias_mitigated_line, baseline_line in zip(bias_mitigated_fd, baseline_fd):
            bias_mitigated_predictions = json.loads(bias_mitigated_line)
            probs = torch.tensor(bias_mitigated_predictions["probs"])
            bias_mitigated_nli(probs)

            baseline_predictions = json.loads(baseline_line)
            probs = torch.tensor(baseline_predictions["probs"])
            baseline_nli(probs)

            if create_diff:
                diff_tool(
                    bias_mitigated_predictions["label"],
                    baseline_predictions["label"],
                    baseline_predictions["tokens"],
                    baseline_dataset_reader._tokenizer.tokenizer,  # type: ignore
                )

    bias_mitigated_metrics = {**bias_mitigated_output_metrics, **(bias_mitigated_nli.get_metric())}
    metrics_json = json.dumps(bias_mitigated_metrics, indent=2)
    if args.bias_mitigated_output_file:
        # write all metrics to output file
        # don't use dump_metrics() because want to log regardless
        with open(args.bias_mitigated_output_file, "w") as fd:
            fd.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    baseline_metrics = {**baseline_output_metrics, **(baseline_nli.get_metric())}
    metrics_json = json.dumps(baseline_metrics, indent=2)
    if args.baseline_output_file:
        # write all metrics to output file
        # don't use dump_metrics() because want to log regardless
        with open(args.baseline_output_file, "w") as fd:
            fd.write(metrics_json)
    logger.info("Metrics: %s", metrics_json)

    if create_diff:
        diff = diff_tool.get_diff()
        diff_json = json.dumps(diff, indent=2)
        if args.predictions_diff_output_file:
            with open(args.predictions_diff_output_file, "w") as fd:
                fd.write(diff_json)
        logger.info("Predictions diff: %s", diff_json)

    logger.info("Finished evaluating.")

    return bias_mitigated_metrics, baseline_metrics
```

This subcommand can be run using: `allennlp evaluate-bias-mitigation /tmp/snli/model.tar.gz https://storage.googleapis.com/allennlp-public-models/snli-roberta.2021-03-11.tar.gz https://storage.googleapis.com/allennlp-public-models/binary-gender-bias-mitigated-snli-dataset.jsonl 32 --bias-mitigated-output-file bm.txt --baseline-output-file bl.txt --predictions-diff-output-file diff.txt`.

4. Inspecting `bm.txt` and `bl.txt` will reveal the following `NaturalLanguageInference` scores for the bias-mitigated, finetuned model and baseline, pretrained model:

```python
"bias_mitigated_model_metrics": {
  "net_neutral": 0.6417539715766907,
  "fraction_neutral": 0.7002295255661011,
  "threshold_0.5": 0.6902161836624146,
  "threshold_0.7": 0.49243637919425964
}, 
"baseline_model_metrics": {
  "net_neutral": 0.49562665820121765,
  "fraction_neutral": 0.5068705677986145,
  "threshold_0.5": 0.47600528597831726,
  "threshold_0.7": 0.3036800026893616
}
```
The `NaturalLanguageInference` scores for the bias-mitigated, finetuned model are much closer to 1 than they are for the baseline, pretrained model, which suggests lower binary gender bias with respect to occupations, as bias will result in a higher probability of entailment or contradiction. `diff.txt` allows us to see specific examples for which the bias-mitigated, finetuned model chose neutral but the baseline, pretrained model didn't:

```python
{
    "sentence_pair": "<s>An accountant can afford a computer.</s></s>A gentleman can afford a computer.</s><pad><pad><pad><pad>",
    "bias_mitigated_label": "neutral",
    "baseline_label": "entailment"
}
```

We have made the [bias-mitigated, finetuned large RoBERTA model for SNLI](https://storage.googleapis.com/allennlp-public-models/binary-gender-bias-mitigated-snli-roberta.2021-05-20.tar.gz) publicly available with a [model card](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/modelcards/pair-classification-binary-gender-bias-mitigated-roberta-snli.json). You can play around with the model on the [AllenNLP demo](https://demo.allennlp.org/textual-entailment/bin-gender-bias-mitigated-roberta-snli).

</exercise>

<exercise id="6" title="Adversarial Bias Mitigation">

In recent years, adversarial networks have become a popular strategy for bias mitigation. This section explores how to adversarially mitigate biases in models using the [`AdversarialBiasMitigator`](http://docs.allennlp.org/main/api/fairness/adversarial_bias_mitigator/#adversarialbiasmitigator) wrapper provided by `allennlp.fairness`. The discussion below draws from: 

1. Zhang, B.H., Lemoine, B., & Mitchell, M. (2018). [Mitigating Unwanted Biases with Adversarial Learning](https://api.semanticscholar.org/CorpusID:9424845). Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society.

2. Zaldivar, A., Hutchinson, B., Lemoine, B., Zhang, B.H., & Mitchell, M. (2018). [Mitigating Unwanted Biases in Word Embeddings with Adversarial Learning](https://colab.research.google.com/notebooks/ml_fairness/adversarial_debiasing.ipynb) colab notebook.

Adversarial networks mitigate biases based on the idea that predicting an outcome Y given an input X should ideally be independent of some protected variable Z. Informally, "knowing Y would not help
you predict Z any better than chance" [2]. This can be achieved using two networks in a series, where the first (known as the predictor) attempts to predict Y using X as input, and the second (known as the adversary) attempts to use the predicted value of Y to recover Z. Please refer to Figure 1 from [Mitigating Unwanted Biases with Adversarial Learning](https://api.semanticscholar.org/CorpusID:9424845) for a diagram of the setup. Ideally, we would like the predictor to predict Y without permitting the adversary to predict Z any better than chance.

![adversarial_figure_1](/part3/fairness/adversarial_figure_1.png)

For common NLP tasks, it's usually clear what X and Y are, but Z is not always available. However, we can construct our own Z by [2]:

1. computing a bias direction (e.g. for binary gender)

2. computing the inner product of static sentence embeddings (mean pooling of static word embeddings) and the bias direction

Like in the previous section, we detail how to apply `AdversarialBiasMitigator` to reduce binary gender bias in [AllenNLP's public large RoBERTA model for SNLI](https://storage.googleapis.com/allennlp-public-models/snli-roberta.2021-03-11.tar.gz), directly from a config file.

1. `AdversarialBiasMitigator` works by chaining an adversary of your choice to the pretrained RoBERTA model, i.e. the predictor, during finetuning. In our example, we use a simple, linear [`FeedForwardRegressionAdversary`](http://docs.allennlp.org/main/api/fairness/adversarial_bias_mitigator/#feedforwardregressionadversary). The adversary takes the predictor's predictions (that is, probabilities of entailment, contradiction, neutrality) as input and attempts to recover the coefficient of the static embedding of the predictor's input sentences in a binary gender [bias direction](http://docs.allennlp.org/main/api/fairness/bias_direction/) of your choice. 

`AdversarialBiasMitigator` works with any Transformer predictor that has a static embedding layer. The relationships between (1) the number of epochs of finetuning, (2) the number of layers in the predictor, (3) the type of bias direction used, and the efficacy of bias mitigation is a promising direction for future research. While finetuning for more epochs with `AdversarialBiasMitigator` might push the predictor to depend less on gender, it could also cause the predictor to overfit to spurious correlations and/or discover proxies for gender that are not contained within the gender bias direction. Furthermore, it is likely harder to mitigate biases for predictors with a larger number of layers, as they are capable of learning extremely complex mappings and achieving convergence in large adversarial networks is tricky (this is further discussed in <b>Pitfalls of Adversarial Bias Mitigation</b> below). Finally, how well `AdversarialBiasMitigator` works strongly depends on the finetuning data and predictor architecture. You should not assume that using a `AdversarialBiasMitigator` resolves any particular bias issue in a predictor. There is published literature suggesting that it should help (without noticeably compromising predictor performance on the test set), but that needs to be evaluated for any actual downstream use.

2. While the adversary's parameters are updated normally during finetuning, the predictor's parameters are updated such that the predictor will not aid the adversary and will make it more difficult for the adversary to recover binary gender bias. Formally, we update the predictor's parameters according to Equation 1 from [Mitigating Unwanted Biases with Adversarial Learning](https://api.semanticscholar.org/CorpusID:9424845). The projection term in Equation 1 ensures that the predictor doesn't accidentally aid the adversary and the final term in Equation 1 serves to update the predictor's parameters to increase the adversary's loss.

![adversarial_equation_1](/part3/fairness/adversarial_equation_1.png)

We achieve this non-traditional parameter update scheme in AllenNLP using an [`on_backward`](http://docs.allennlp.org/main/api/training/callbacks/callback/#on_backward) trainer callback. An [`on_backward`](http://docs.allennlp.org/main/api/training/callbacks/backward/) trainer callback performs custom backpropagation and allows for gradient manipulation; `on_backward` callback(s) will be executed in lieu of the default `loss.backward()` called by `GradientDescentTrainer`.

We provide the [`AdversarialBiasMitigatorBackwardCallback`](http://docs.allennlp.org/main/api/fairness/adversarial_bias_mitigator/#adversarialbiasmitigatorbackwardcallback) to perform the non-traditional parameter updates required by adversarial bias mitigation.

```python
@TrainerCallback.register("adversarial_bias_mitigator_backward")
class AdversarialBiasMitigatorBackwardCallback(TrainerCallback):
    """
    Performs backpropagation for adversarial bias mitigation.
    While the adversary's parameter updates are computed normally,
    the predictor's parameters are updated such that the predictor
    will not aid the adversary and will make it more difficult
    for the adversary to recover protected variables.

    !!! Note
        Intended to be used with `AdversarialBiasMitigator`.
        trainer.model is expected to have `predictor` and `adversary` data members.

    # Parameters

    adversary_loss_weight : `float`, optional (default = `1.0`)
        Quantifies how difficult predictor makes it for adversary to recover protected variables.
    """

    def __init__(self, serialization_dir: str, adversary_loss_weight: float = 1.0) -> None:
        super().__init__(serialization_dir)
        self.adversary_loss_weight = adversary_loss_weight

    def on_backward(
        self,
        trainer: GradientDescentTrainer,
        batch_outputs: Dict[str, torch.Tensor],
        backward_called: bool,
        **kwargs,
    ) -> bool:
        if backward_called:
            raise OnBackwardException()

        if not hasattr(trainer.model, "predictor") or not hasattr(trainer.model, "adversary"):
            raise ConfigurationError(
                "Model is expected to have `predictor` and `adversary` data members."
            )

        trainer.optimizer.zero_grad()
        # `retain_graph=True` prevents computation graph from being erased
        batch_outputs["adversary_loss"].backward(retain_graph=True)
        # trainer.model is expected to have `predictor` and `adversary` data members
        adversary_loss_grad = {
            name: param.grad.clone()
            for name, param in trainer.model.predictor.named_parameters()
            if param.grad is not None
        }

        trainer.model.predictor.zero_grad()
        batch_outputs["loss"].backward()

        with torch.no_grad():
            for name, param in trainer.model.predictor.named_parameters():
                if param.grad is not None:
                    unit_adversary_loss_grad = adversary_loss_grad[name] / torch.linalg.norm(
                        adversary_loss_grad[name]
                    )
                    # prevent predictor from accidentally aiding adversary
                    # by removing projection of predictor loss grad onto adversary loss grad
                    param.grad -= (
                        (param.grad * unit_adversary_loss_grad) * unit_adversary_loss_grad
                    ).sum()
                    # make it difficult for adversary to recover protected variables
                    param.grad -= self.adversary_loss_weight * adversary_loss_grad[name]

        # remove adversary_loss from computation graph
        batch_outputs["adversary_loss"] = batch_outputs["adversary_loss"].detach()
        return True
```

Below is an [example config file](https://github.com/allenai/allennlp-models/blob/main/training_config/pair_classification/adversarial_binary_gender_bias_mitigated_snli_roberta.jsonnet) for finetuning a pretrained RoBERTA model for SNLI using `AdversarialBiasMitigator` with a two-means bias direction. As mentioned previously, `AdversarialBiasMitigator` simply wraps the predictor and adversary. `two_means` requires a `seed_word_pairs_file` and tokenizer to tokenize the words in said file. `seed_word_pairs_file` must follow the format in the example: `[["woman", "man"], ["girl", "boy"], ["she", "he"], ...]`. And that's it!

```python
local transformer_model = "roberta-large";
local transformer_dim = 1024;

{
  "dataset_reader": {
    "type": "snli",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "add_special_tokens": false
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": 512
      }
    }
  },
  "train_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_train.jsonl",
  "validation_data_path": "https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_dev.jsonl",
  "model": {
    "type": "adversarial_bias_mitigator",
    "predictor": {
      "_pretrained": {
        "archive_file": "https://storage.googleapis.com/allennlp-public-models/snli-roberta.2021-03-11.tar.gz",
        "module_path": "",
        "freeze": false
      }
    },
    "adversary": {
        "type": "feedforward_regression_adversary",
        "feedforward": {
            "input_dim": 3,
            "num_layers": 1,
            "hidden_dims": 1,
            "activations": "linear"
        }
    },
    "bias_direction": {
        "type": "two_means",
        "seed_word_pairs_file": "https://raw.githubusercontent.com/tolga-b/debiaswe/4c3fa843ffff45115c43fe112d4283c91d225c09/data/definitional_pairs.json",
        "tokenizer": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
          "max_length": 512
        }
    },
    "predictor_output_key": "probs"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "batch_size" : 32
    }
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "+accuracy",
    "callbacks": [
        "adversarial_bias_mitigator_backward"
    ],
    "optimizer": {
        "type": "multi",
        "optimizers": {
            "predictor": {
                "type": "adam",
                "lr": 1e-5
            },
            "adversary": {
                "type": "adam",
                "lr": 1e-5
            },
            "default": {
                "type": "adam",
                "lr": 1e-5
            }
        },
        "parameter_groups": [
            [
                [
                    "^predictor"
                ],
                {
                    "optimizer_name": "predictor"
                }
            ],
            [
                [
                    "^adversary"
                ],
                {
                    "optimizer_name": "adversary"
                }
            ]
        ]
    }
  }
}
```

You can run this config file with the following CLI command: `allennlp train config.jsonnet --include-package allennlp_models -s /tmp/snli`.

3. After ten epochs of finetuning, we can observe a considerable reduction in binary gender bias with respect to occupations, as measured by the `NaturalLanguageInference` bias metric on a [large dataset of neutrally-constructed textual entailment sentence pairs](https://storage.googleapis.com/allennlp-public-models/binary-gender-bias-mitigated-snli-dataset.jsonl). Even better, this comes without a noticeable difference in performance on the test set!

4. Using the `EvaluateBiasMitigation` subcommand written in the previous section, we can compare the `NaturalLanguageInference` scores for the adversarial-bias-mitigated, finetuned predictor and baseline, pretrained predictor, after 1 and 10 epochs of finetuning:

After 1 epoch of finetuning (model with highest accuracy on validation set):
```python
"adversarial_bias_mitigated_predictor_metrics": {
    "net_neutral": 0.4410316309540555,
    "fraction_neutral": 0.4481893218322427,
    "threshold_0.5": 0.4315516764161544,
    "threshold_0.7": 0.3035287155463018

},
"baseline_predictor_metrics": {
    "net_neutral": 0.49562667062645066,
    "fraction_neutral": 0.506870600337101,
    "threshold_0.5": 0.47600531264458984,
    "threshold_0.7": 0.30368001850750215
}
```

After 10 epochs of finetuning:
```python
"adversarial_bias_mitigated_predictor_metrics": {
    "net_neutral": 0.613096454815352,
    "fraction_neutral": 0.6704967487937075,
    "threshold_0.5": 0.6637061892722586,
    "threshold_0.7": 0.49490217463150243
},
"baseline_predictor_metrics": {
    "net_neutral": 0.49562667062645066,
    "fraction_neutral": 0.506870600337101,
    "threshold_0.5": 0.47600531264458984,
    "threshold_0.7": 0.30368001850750215
}
```

The `NaturalLanguageInference` scores for the adversarial-bias-mitigated, 10-epoch-finetuned predictor are much closer to 1 than they are for the 1-epoch-finetuned predictor and the baseline, pretrained predictor. This suggests lower binary gender bias with respect to occupations, as bias will result in a higher probability of entailment or contradiction. However, the bias-mitigated, finetuned model from the previous section boasts slightly higher `NaturalLanguageInference` scores for this downstream task.

Unlike for the bias-mitigated model from the previous section, the adversarial-bias-mitigated predictor with the highest validation accuracy (in this case, 1 epoch of finetuning) does not demonstrate a significant reduction in binary gender bias, and the adversarial-bias-mitigated predictor seems to benefit from more epochs of finetuning. 

`diff.txt` allows us to see specific examples for which the adversarial-bias-mitigated, 10-epoch-finetuned predictor chose neutral but the baseline, pretrained model didn't:

```python
{
    "sentence_pair": "<s>A zoologist spoke to an adult.</s></s>A woman spoke to an adult.</s><pad><pad><pad><pad>",
    "adversarial_bias_mitigated_label": "neutral",
    "baseline_label": "entailment"
}
```

We have made the [adversarial-bias-mitigated, finetuned large RoBERTA model for SNLI](https://storage.googleapis.com/allennlp-public-models/adversarial-binary-gender-bias-mitigated-snli-roberta.2021-06-17.tar.gz) publicly available with a [model card](https://github.com/allenai/allennlp-models/blob/main/allennlp_models/modelcards/pair-classification-adversarial-binary-gender-bias-mitigated-roberta-snli.json). You can play around with the model on the [AllenNLP demo](https://demo.allennlp.org/textual-entailment/adv-bin-gen-bias-mitigated-roberta-snli).

## Pitfalls of Adversarial Bias Mitigation

Adversarial bias mitigation does have some pitfalls. For one, both the predictor and adversary are black boxes, which makes it uninterpretable in contrast to the bias mitigators described in previous sections. Furthermore, while [1] derives theoretical guarantees for adversarial bias mitigation, the proof relies on the strong, and impractical, assumption that the loss functions of the predictor and adversary are convex with respect to their parameters. Finally, achieving convergence in adversarial networks is tricky in practice, especially for large, contextual language models; some tips for training adversarial networks include to [2]:

1. lower the step size of both the predictor and adversary to train both models slowly to avoid parameters diverging,

2. initialize the parameters of the adversary to be small to avoid the predictor overfitting against a sub-optimal adversary,

3. increase the adversary’s learning rate to prevent divergence if the predictor is too good at hiding the protected variable from the adversary,

4. use a pretrained predictor to increase the likelihood of convergence. 

</exercise>

<exercise id="7" title="Next Steps">

**Friendly reminder:** While the fairness and bias mitigation tools provided by `allennlp.fairness` help reduce and control biases and increase how equitably models perform, they do <em>not</em> completely remove biases or make models entirely fair. Bias mitigation must go hand-in-hand with the auditing of real-world model performance to ensure that humans are not unfairly impacted by the model's decisions.

We're excited to see your contributions to the Fairness module! Some great additions would be:
- [dataset and model bias amplification metrics](https://api.semanticscholar.org/CorpusID:195847929)
- training-time and post-processing algorithms for fairness without demographics, e.g. [Fairness Without Demographics in Repeated Loss Minimization](https://api.semanticscholar.org/CorpusID:49343170), [Fairness without Demographics through Adversarially Reweighted Learning](https://api.semanticscholar.org/CorpusID:219980622)
</exercise>
