---
title: 'Interpret Module'
description: A practical guid into AllenNLP Interpret Module
author: Zeyu Liu 
type: chapter
---

<textblock>
Essentially, our methods in the `allennlp.interpret` module provide the interface to look into your "black-box" models. 
This tutorial tries to be self-contained and will start with a simple discussion of why we need this module. 
Later, we will provide more in-and-out details on our provided methods.
</textblock>

<exercise id="1" title="Why do we need interpretation methods?">

## Black v.s. white box models: A tricky trade-off

A simple example of a white-box and a black-box model is as follows:

|   Whitebox    |          Blackbox           |
| :------------: | :--------------------------: |
| $y= W_2(W_1x)$ | $y= W_2\text{Sigmoid}(W_1x)$ |

By using the more complicated and more expressive model blackbox model, 
we could capture both linear and non-linear relationship between $x$ and $y$ or features in $x$ at different levels 
(e.g. at different layer of neurons of the neural network). In NLP, the inputs (e.g. tokens) are discrete, and so are 
represented by continuous embeddings (treated as part of model parameter). 
Introducing a non-linearity into the embeddings helps us to learn better dense representations with arguably more 
information encoded in comparison to high dimensional one-hot encodings of a token. 
Remember, if there are no non-linearities (like the whitebox case above), 
the model is essentially $W x = W_2 (W_1 x)$ --- adding more parameters doesn't allows us to capture more complicated relationships.


More benefits could be discussed about using blackbox model. However, as always, there's no free lunch. 
Along with the blackbox model, there is a price --- the model is not easily interpreted by a human. 
It is usually very hard to understand the association learned by the model's weights, especially when the model
 consists of many non-linear functions, one on top of another, like with transformer models. 


One could argue that we could still gauge the effect of input or weight (of a blackbox model) by tracing how it 
leads to the model's final prediction. This is certainly true, but complicated by the fact that 
**each additional non-linear function** makes the model more and more inaccessible to users, including both laymen
 and experts. In addition, at their current state blackbox NLP systems are being integrated more and more to human
  daily life. Towards that end, trust in these system is vital and humans tends to distrust things they don't
   understand. Such a pressing need has spurred a bulk of research in interpretability / explanability in NLP system.
    Currently, `allennlp.interpret` supports three types of interpretability methods. In order to explain these, 
    we'll start by describing the learning problem of interest.

</exercise>

<exercise id="2" title="An overview on `allennlp.interpret`">

## The learning problem

For a supervised learning problem, we want to learn a prediction function from an input space $\mathcal{X}$ (e.g. human sentences in text form), to an output space $\mathcal{Y}$ (e.g. labels). We are given $n$ training points $\{\cdots, z_i = (x_i, y_i), \cdots \}_{i=1}^n$. For a datapoint point $z = (x, y)$ and parameters $\theta$, let $L(z; \theta)$ be the loss, and let $\frac1n \sum_{i=1}^n L(z_i; \theta)$ be the empirical risk. Our empirical risk minimizer is given by  $\hat{\theta} = \text{argmin}_{\theta} \frac1n \sum_{i=1}^n L(z_i; \theta)$. In the following part, we will ground our discussion in **sentiment analysis**.


In this **sentiment analysis** task, let's say we have the following input-output pair:
$$
x: \text{a very well-made, funny and entertaining picture.}\\
y: \text{Positive (1)}
$$
The input token at the $t$th position is represented by a $d$-dimensional vector (called word embedding), denoted as $x^{(t)} \in \mathbb{R}^d$. E.g., $x^{(1)} = \text{word embedding of "very"}$. The function we want to learn is essentially a conditional probability distribution ---  $p(y \mid x)$. Given the sentence, the model gives its best guess about the sentiment of the sentence --- positive or negative.

![input_reduction_prediction](/part3/interpret/input_reduction_prediction.png)


## Perspective 1: Interpreting test examples

Currently, there are two prevalent techniques for interpreting test examples: **saliency maps** and **adversarial attacks**. 

### Method 1: Saliency maps

Assuming the model and its loss function are differentiable, we are able to get the gradient with 
respect to each input's embedding, i.e. $\frac{\partial p(y \mid x)}{\partial x^{(t)}} \in \mathbb{R}^d$. 
**Note:** this instance is not labeled, but this doesn't matter. We are interested in why model is more confident 
about it's prediction, i.e., the label with max probability, so we just treat the model's prediciton as a *"gold label"* 
and use the training objective to calcualte the gradient.


By aggregating the gradients of each embedding, we are able to calculate a score for each input word. 
This score represents how sensitive the model is to this word when predicting the label. 
Thus, in this sense, this score represent how **important** this word is and sheds light on how the model made its decision. 
The reason for the term "map" is that a very intuitive method for visualizing saliency is through a 1-dimentional heatmap.
 An illustrative visualization is:

![saliency_map_example](/part3/interpret/saliency_map_example.png)

We only highlight the top 3 most important words.

### Method 2: Adversarial attacks

Saliency maps is based on the following intuition:
 given the current prediction task ---  $p(\text{positive} \mid \text{sentence})$ --- 
 what if we perturb the input embedding by $\varepsilon \rightarrow 0$ in the direction specified 
 by its gradient $\frac{\partial p(y \mid x)}{\partial x_i}$? 
 Another more "higher" level perturbation is this: instead of perturbing a token's embedding,
  what if we perturb the tokens themselves?

There are two ways to go about this: replacing words to change the model’s prediction ([HotFlip](https://arxiv.org/abs/1712.06751))
 and removing words to maintain the model’s prediction ([Input Reduction](https://arxiv.org/abs/1804.07781)). 
 For illustrative purpose, we discuss Input Reduction. 



In Input Reduction, we keep iteratively removing the token with the smallest "gradient-based score"
 (e.g. how we calculate the saliency maps) from the sentence. 
 We stop at the last step before the decision flips --- from positive sentiment to negative,
  or vice versa. Since removing the tokens doesn't change the predicted result, 
  we can postulate that those tokens are not significant in helping to make the prediction.

An example is shown as follow:

![input_reduction_example](/part3/interpret/input_reduction_example.png)


## Perspective 2: Interpreting train examples

Now, to conclude a bit, two aforementioned methods made purturbation on **individual** test examples 
to understand how model use information in the test example to make prediction. 
There is another diagnal dimension for interpreting the model --- how train set affects the model's prediction on a test example? 
What if we "purturb" one of the training example, what changes would it make on the model's performance on a test example? 



This can be answered by using an **influence function**. 
The idea is to compute the parameter change if we up-weight the loss for a particular data point 
--- $L(z, \theta)$ --- by a small amount $\varepsilon \rightarrow 0$. 
In particular, we look at how doing this affects the model's performance on a particular test data point --- $z_{\text{test}}$? 
These types of methods are also called  **example-based**. To estimate the influence,
 [Koh and Liang (2017)](https://arxiv.org/pdf/1703.04730.pdf) propose to estimate the influence score with the following formula.

![influence_function_formular](/part3/interpret/influence_function_formular.png) 

### A TL;DR understanding

The second and third items together (and the nagative sign), i.e. $-H_{\hat{\theta}}^{-1} \nabla_{\theta}L(z, \hat{\theta})$, 
measures, **hypothetically**, what if we upweight the loss for training example $z$ -- $L(z, \hat{\theta})$ -- by $\varepsilon$, 
how much **will** it affect our obtained parameter $\hat{\theta}$; 
its form is derived from a quadratic approximation to the training loss around $\hat{\theta}$, 
see more details in [Koh and Liang (2017)](https://arxiv.org/pdf/1703.04730.pdf). 

The first term, i.e. $\nabla_{\theta}L(z_{\text{test}}, \hat{\theta})$, 
came from applying chain rules and rein in the effect of how upweighting $L(z, \hat{\theta})$ will influence loss at test point $z_{\text{test}}$.

An contrastive example, in sentiment analysis, to compare saliency map and influence function is as follows. 

![contrastive_example](/part3/interpret/contrastive_example.png)


</exercise>

<exercise id="3" title="Sliancy Map">
Saliency maps interact with model through the input tokens' embeddings. 
It essentially will return a normalized score (calculated from embedding's gradient) that represents 
how sensitive the model is towards the change in the tokens' embeddings. 
The higher the score, the more sensitive the model is. By "normalized", we mean, 
after aggregation, the scores are weighted to make sure the scores sum up to 1. 
The reason why it is called "maps" is rooted from computer vision. 
Since the input is a 2-D image (cmp with 1-D string in NLP), the calculated gradient (w.r.t the input image) 
thus is also 2-D; therefore, it's natural to visualize the gradient as a new 
image and treat it as a *map* to tell which part of the input is *salient* in giving the model's final prediction. 



Note that this score is aggregated from embeddings' gradients; 
so the scores could tell us an overall difference between tokens, conditioned on the aggregation methods.

Depending on how the gradients were collected from the model (w.r.t. the test example), 
we could have different methods --- **Vanilla Gradient** (`SimpleGradient`), **Integrated Gradient** (`IntegratedGradient`), 
**Smooth Gradient** (`SmoothGradient`). However, all of them share the same interface. 
**To swap different methods, simply replacing the class name will achieve what you want**.
 Now the code is written as follows.



```python
from allennlp.interpret.saliency_interpreters import SimpleGradient
from allennlp.predictors import Predictor, TextClassifierPredictor
from allennlp.models.archival import load_archive

inputs = {"sentence": "It was the ending that I hated"}
archive = load_archive("path/to/your/model.tar.gz")
predictor = Predictor.from_archive(archive)
interpreter = SimpleGradient(predictor)
interpretation = interpreter.saliency_interpret_from_json(inputs)
```

`path/to/your/model.tar.gz` is usually self-explainatory after you use `allennlp train` or 
equivalent code to train a model with AllenNLP. It is simply the archive file that you get after training.

If you are interested in experimenting with one of the models supported by `allennlp-models`, 
please replace the code above related to `archive` and `predictor` with the following.

```python
from allennlp_models.pretrained import load_predictor

predictor = load_predictor("<predictor of your choice>")
```

In its current state, our saliency interpreters work only with models designed for label prediciton tasks 
(e.g. sentiment analysis and NER). For generation tasks that involve sampling, e.g., beam search, we don't guarantee that these will work.



# A bit more in-depth:

Let $x$ represent the embedded sentence, where $x^{(t)}$ represents the embedding of $t$th token. 
Also let $p$ represent the objective function of the task, and assume that the model is some differentiable blackbox model.

### 1. Vanilla Gradient

Probably the easiest way to collect gradient information for each token is taking $\frac{\partial p}{\partial x^{(t)}}$. 
This tells you how sensitive the model is with respect to the input embedding.

Then, to get a score for each tokens, we choose to do $\|x^{(t)} \odot \frac{\partial p}{\partial x^{(t)}}\|_1$, 
where $\odot$ denotes element-wise multiplication, and $\|\cdot\|_1$ denotes $L$-1 norm. 
Here, the reason we do the element-wise multiplication with the input $x^{(t)}$ is to account for two facts:  

1. Some entries in $x^{(t)}$ might have small values, but large gradient; however, those entries should, in general, have small effect on producing the final result. 
2. Some entries in $x^{(t)}$ might have large values in producing the final results, but small gradient. Those entries usually also means small effect on producing the final results, because small gradient implies the model is insensitive to the changes in those entries.

Therefore, we want some "average-out" operation and our choice is the element-wise multiplication. 
And finally we will get a list of scores $\{s^{(t)}\}_{t=1}^{|x|}$; 
for normalization, we just do $w^{(t)} = \frac{s^{(t)}}{\sum_u s^{(u)}}$.



### 2. Integrated Gradient

Another intuitive way to think about the purpturbation on the gradient is "*counterfactual*" 
--- what if the input is not $x$ but some sort of *baseline* $x'$? 
In case of computer vision, what if your received image $x'$ is all black? 
In case of natural language processing (NLP), what if the word embeddings $x'$ are all-zero vectors? 

Those baselines are chosen so that they convey a complete absence of signals. 
A black image should be a self-explanatory example; for the all-zero embedding case, 
it is because researchers generally observe unimportant words tend to have small norms, 
and so in the limit, "unimportance" corresponds to an all-zero baseline.

The Integrated Gradient of $x$ is defined as the **path integral of the gradients along 
the straight line path from the baseline $x'$ to the input $x$**. 
Compared with Vanilla Gradient, this methods has several appealing and theoretically guaranteed properties:

1. If for every $x$ and $x'$ that differ in one feature but have different predictions then the differing feature should be given a non-zero attribution. 

2. If the function implemented by the deep network does not depend (mathematically) on some variable, then the attribution to that variable is always zero.

3. Attributions from this methods *add up* to the difference between the output of $p$ at the input $x$ and $x'$

   

The integrated gradient along the $i$th dimension for an input $x$ and $x'$ is written as
$$
(x_i - x'_i) \times \int_{\alpha=0}^1 \frac{\partial p(x' + \alpha \times (x - x'))}{\partial x_i} \; d\alpha
$$
This integral doesn't have a closed-form solution, and thus need to be approximated. We go with the popular choice of [Riemann approximation](https://en.wikipedia.org/wiki/Riemann_sum), as followed. 
$$
(x_i - x_i') \times \sum_{k=1}^m \frac{\partial p(x' + \frac{k}{m} \times (x - x'))}{\partial x_i} \times \frac1m
$$
Here $m$ is the number of steps in the Riemman approximation of the integral. The calculation for other dimension could be "parallelized" in tensor operations, which is handled by package like PyTorch. What we essentially do in our module is what the following python-like pseudo-code specifies

```python
# Given: input_embedding
m = 10
ig_grads = 0
for k_over_m in numpy.linspace(0, 1.0, num=m):
  # getting the second term 
  input_embedding *= k_over_m
  grads = self.predictor.get_gradients(input_embedding) # calling torch.autograd()
  #  accumulating for summation over k
  ig_grads += grads

# multiply by 1/m
ig_grads /= m
```

For more detailed discussion and proof, we refer reader to the original [paper](https://arxiv.org/abs/1703.01365).



### 3. Smooth Gradient

Another very simple technique is adopted from Computer Vision, where it aims at producing **sharper** saliency maps. 
This technique might not work as well in NLP, because each word in sentence might be equivalent to hundreds of 
pixels in image input  --- word embedding has hundreds of entries --- but we want a score for each word; 
thus, detailed information like gradient of some word embedding's entry will be aggregated out. 

To best convey the idea, we explain the techniques in Computer Vision secnario; 
however, do keep in mind that similar effects *will* appear in NLP, potentially only smaller. 

When the input is an image containing thousands of pixels, it's generally observed that saliency maps produced 
from previous two gradient are often found to be *noisy* --- there are some pixels highlighted by the gradient-based 
techniques and yet incomprehensible to human. In another words, the saliency maps looks blurry due to those noise. 

However, the battle is not lost. It is observed that the partial derivative $\frac{\partial p}{\partial x_i}$ varies 
significantly when researchers make purturbation to $x_i$, where $x_i$ denotes a pixel. 
What's specifically done is researchers have a small random noise $\varepsilon \sim \mathcal{N}(0, 0.01^2)$.
 As a result, $x+\varepsilon$ is indistinguishable to human percetion; however, 
 when researchers set input of the model to be $x+ t \cdot \varepsilon $, where $t\in [0,1]$, 
 researchers found the resulting $\frac{\partial p}{\partial x_i}$ changes significantly against $t$. 

Taking a step back, there is no reason to expect the derivative to vary smoothly with respect to the perturbation, 
because the model we are dealing with is non-convex. 
This means the apparent noise one sees in a sensitivity map may be due to meaningless local variations in the partial derivatives. 
Therefore, one very simple and intuitive way to improve the saliency maps is to 
**take random samples in neighborhood of an input $x$, and average the resulting saliency maps**.
 Mathematically, this means calculating
 
$$
\frac1n \sum_{1}^n \frac{\partial p(x + \mathcal{N}(0, \sigma^2))}{\partial x_i}
$$

where $n$ is the number of samples and $\mathcal{N}$ represents Gaussian noise with standard deviation $\sigma$. 
**Empirically speaking**, generating saliency maps with this additional sampling generates sharper results.

For more detailed discussion and proof, we refer reader to the original [paper](https://arxiv.org/abs/1706.03825).

To adapt this techniques to NLP, see the following schematic code

```python
# Given: input_embedding
n = 10
stdev = 0.01
smooth_grads = 0
for _ in range(n):
  # getting the second term 
  noise = torch.randn(output.shape, device=output.device) * stdev
  input_embedding += noise
  grads = self.predictor.get_gradients(input_embedding) # calling torch.autograd()
  #  accumulating for summation over n
  smooth_grads += grads

# multiply by 1/n
smooth_grads /= n
```

</exercise>

<exercise id="4" title="Attacker">

For the discussion in this section, we would inherit the notation from saliency maps.

Attackers interact with the interested model in a higher granularity. 
Saliency maps is perturbing word embeddings with small $\varepsilon$ and see how much it changes the results of model; 
for the embedding of $t$th token $x^{(t)}$, the effect is measured by the gradient 
$\frac{\partial p}{\partial x^{(t)}} = \lim_{\varepsilon\rightarrow 0} \frac{p(x + \varepsilon \cdot x^{(t)}) - p(x)}{\varepsilon \cdot x^{(t)}}$. 
In comparison, our attackers would change the input at word level --- a much more perceptible level for human. 
Depending on various way to "change inputs", we have two different attackers 
--- **Hopflip**(`Hotflip`; [Ebrahimi et. al](https://arxiv.org/pdf/1712.06751.pdf)) and 
**Input Reduction**(`InputReduction`; [Feng et. al](https://arxiv.org/pdf/1804.07781.pdf))

The way to use our attackers is still fairly simple. 


```python
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.interpret.attackers import InputReduction

inputs = {"sentence": "It was the ending that I hated"}
archive = load_archive("path/to/your/model.tar.gz")
predictor = Predictor.from_archive(archive)
reducer = InputReduction(predictor) # or Hotflip(predictor)
# if it is Hotflip, we need an extra step: reducer.initialize()
reduced = reducer.attack_from_json(inputs, "tokens", "grad_input_1")
```

`path/to/your/model.tar.gz` is your serialization directory after using `allennlp train` 
or equivalent code to train a model with AllenNLP. 

We want to further note the usage of `attck_from_json`. 
Due to the unpredictable structure of an instance, we leave it to the user's 
responsibility to specify the field to attack --`"tokens"`--and the gradient field -- `"grad_input_1"`. 


So far, the attackers work on a small number of tasks for illustrative purposes. 
`InputReduction` works for Named Entity Recognition and Sentiment Analysis, 
and `Hotflip` works for Sentiment Analysis only. 
If you are interested in the underlying work behind each attacker class, you should read on. 


# A bit more in-depth:

### 1. Hotflip attacker

The procedure encapsulated in this class intends to **change the model's prediction as quickly as possible**. 
The motivation is to identify the words that are **important** for the current prediction; this is completed by

1. Given the current sentence $x$, get gradients  $\frac{\partial p}{\partial x^{(t)}}, t = \{1, 2, \cdots, |x|\}$
2. "Flip" one of the current words into a new word to maximize the first-order Taylor approximation of the loss
3. Re-evaluate the model's prediction and stop if prediction changes; otherwise, remove the word and go to step 1.

Now, the current description makes two things vague --- "current word" and the "first-order Taylor approximation of the loss."

#### 1.1 Current word

Due to Step 3, we have a different input sentence $x$ at inspection at every iteration -- 
otherwise, it means no change from the last iteration, and we should exit the loop! 

At the current iteration, we want to ask the question, which word should I change? 
Remember that our goal is to flip the prediction, which aligns well with the intention of the loss function. 
Therefore, the most straightforwards approach is to select the word which has the highest $L$-2 norm
 --- $\|\frac{\partial p}{\partial x^{(t)}}\|_2$ --- among other words at the current iteration. 

Note that we need to re-evaluate both the prediction and the gradient; 
though logically, word at one position could be flipped many times, we enforce the attacker to flip one position only once.

#### 1.2 First-order Taylor approximation

Recall that Taylor expansion or series of a function $f(b)$ around a "center point" $a$ is 


$$
\sum_{n=0}^\infin \frac{f^{(n)}(a)}{n!}(b - a)^n = f(a) + \frac{f'(a)}{1!}(b - a) + \frac{f''(a)}{2!}(b - a)^2 + \dots
$$


Now, in a our specific case, our objective function $p$ is the function $f$ in the equation. 
And the center point $a$ is orignal embedded sentence $x = [w_1, w_2, \cdots, w_{|x|}]$. 
Remember, what we do at every step 2 is to flip one token $w_i$ to some $\hat{w}$ recognized by the model. 
As a result we would have a new sentence $\hat{x} = [w_1, \cdots, w_{i-1}, \hat{w}, w_{i+1},\cdots, w_{|x|}]$. 
So, our new loss $p(\hat{x})$ in Taylor series around $x$ is $\sum_{n=0}^\infin \frac{p^{(n)}(x)}{n!}(\hat{x} - x)^n $. 
A first-order approximation is then $p(\hat{x})\approx \sum_{n=0}^1 \frac{p^{(n)}(x)}{n!}(\hat{x} - x)^n = p(x) + p'(x) \cdot (\hat{x} - x)$. 
To measure how much the loss changes, we just calculate $p(\hat{x}) - p(x)$. With first-order Taylor approximation, we could...



Wait, wait, wait, why do we want to use Taylor approximation? Why not just naively calculate the exact loss every time we flip a word? 
**Computation**. We could re-use the gradient $p'(x)$ with the approximation, which we get from one backward pass. 
Otherwise, we need to have a large number of forward passes. 

With first-order Taylor approximation, we know the loss change $p(\hat{x}) - p(x)$ could be approximated by 
$$
p(x) + p'(x) \cdot (\hat{x} - x) - p(x) = p'(x) \cdot (\hat{x} - x)
$$


#### Misc

##### `reducer.intialize()`?

The need for `reducer.initialize()` for `Hotflip` is that sometimes the vector that finally represents a 
token unit (to the model) has some projection layers after the embedding layer. 
For example, ELMo. In those cases, we need to create a new "embedding" matrix representing "words" in the model's eyes. 

##### Character flip

Our Hotflip attackers also support flipping at the character level. You would do something like

```python
predictor._dataset_reader._token_indexers["chars"] = TokenCharactersIndexer(min_padding_length=1)
predictor._model._text_field_embedder._token_embedders["chars"] = # Some subclass of `TokenEmbedder`
```

##### Two other parameters for `attack_from_json`

Users could also specify `target` and `ignore_tokens` to the function.

If `target`is given, instead of just trying to change a model's prediction from what it currently predicts, 
we try to change it to a specific target value. This needs to be a dictionary because it needs to specify the 
field name and target value. For example, for a *masked LM*, this would be something like `{"tokens": ["she"]}`, 
because `"words"` is the field name, there is one mask
token, and we want to change the prediction from whatever it was to `"she"`.

`ignore_tokens` specify tokens that won't be flipped, e.g. special tokens representing out-of-vocabulary and padding tokens, etc. 


### 2. Input Reduction Attacker

This attacker changes the input from an opposite direction --- remove as many "unimportant words" 
as possible till the decision changes. Compared with Step 2 in `Hotflip`, `InputReduction` do the following:

**Step 2**: use $\|x^{(t)} \odot \frac{\partial p}{\partial x^{(t)}}\|_2, t = \{1, 2, \cdots, |x|\}$ 
to approximately measure a token's importance --- the higher the more important. 
At the current step, we **remove** the most unimportant word (`Hopflip` **substitutes**). 
Then go to the same Step 3 in `HotFlip`.

This should sound worried to you because this is just a greedy search, which usually leads to a sub-optimal solution. 
Therefore, to find the shortest possible reduced inputs, **beam search** is applied. 
We limit the removal to the $k$ least important words, where $k$ is the beam size. 
Then, we decrease the beam size as the remaining input is shortened, i.e., 
$\text{beam\_width}_{\text{cur}} = \max(1, \min(k, L − 3))$, where $L$ denotes number of tokens currently remained.

</exercise>


<exercise id="5" title="Influence Function">

Recall from the above. Saliency maps and attackers focus on figuring out which part of the **test input** 
is most important for the model to make its prediction. 
Both of them assume a model already trained on some set of training examples. 
In contrast, the influence function puts the test input aside and attempts to answer the question 
--- **how each training example affects model prediction on the test example? Which one is more important? Could we score them?**

We introduce a new class, `InfluenceInterpreter`, towards that end. 
Currently, we only migrate what researchers did by naively applying influence function ([Koh and Liang (2017)](https://arxiv.org/pdf/1703.04730.pdf)) 
to simple tasks like natural language inference ---  `SimpleInfluence`. Its usage will be similar to how you use saliency maps and attackers.

```python
from allennlp.interpret.influence_interpreters import SimpleInfluence
from allennlp_models.pretrained import load_predictor

from allennlp_models.pair_classification.predictors.textual_entailment import (
    TextualEntailmentPredictor,
)
from allennlp.models.archival import load_archive

archive_file = "../allennlp-models/mnli_bert_output/model.tar.gz"
train_file = "../allennlp-models/data/multinli_1.0_train.jsonl"
test_file = "../allennlp-models/data/multinli_1.0_dev_mismatched.jsonl"

archive = load_archive(archive_file)
predictor = Predictor.from_archive(archive)
dataset_reader = roberta_nli_predictor._dataset_reader
simple_if = SimpleInfluence(
    predictor,
    train_file,
    test_file,
    dataset_reader,
  	k=20
)
simple_if.calculate_inflence_and_save("path/to/your/output.jsonl")
```

For memory-efficient reason, we made the decision to only save training examples with $k$ **highest** and 
$k$ **lowest** scores. At the current stage of development, this model is tested to work on classification task like 
natural language inference. Each line in the output file, it will look like (just for illustrative purpose):

```json
{"test_instance": {"premise": "The new rights are nice enough", 
                   "hypothesis": "Everyone really likes the newest benefits", 
                   "label": "neutral",
                   "loss": xxx},
 "top_20_train_instances": [{train_instance1}, ...., {train_instance20}], 
 "bottom_20_train_instances": [{train_instance1}, ...., {train_instance20}] 
}
```

Note each training instance has the same structure as the test instance. 
For saving instances, we have introduced a new function, `human_readable_dict()` to `Instance`, 
so don't worry if your instances have a different structure. 



# A bit more in-depth

Remember from the previous introduction, the "influence" of a training example on one test example 
is calculated by the following equation, a.k.a., **influence function**:

![influence_function_formular](/Users/leoliu/My/proj/allennlp-internal/tutorial/influence_function_formular.png)

where 
$$
H_{\hat{\theta}} = \frac1n \sum_{i=1}^n \nabla_{\theta}^2 L(z_i, \hat{\theta})
$$
is the average over training examples' Hessians. To make things worse, we need to calculate its inverse. 
Despite this heavy computation, an algorithm called **LiSSA**, which requires some extra sampling from the training set,
 can approximated the inverse. 
 We refer the user to **Stochastic estimation** paragraph in [Koh and Liang (2017)](https://arxiv.org/pdf/1703.04730.pdf) 
 for perfect summarization of high-level idea and [ABH (2016)](https://arxiv.org/pdf/1602.03943.pdf) for theoretical proofs of correctness. 

LiSSA algorithm requires some hyperparameters, which we also support for interested users to try out. 
According to various literature ([Koh and Liang (2017)](https://arxiv.org/pdf/1703.04730.pdf), 
[Han et al. (2020)](https://www.aclweb.org/anthology/2020.acl-main.492.pdf), 
[Guo et al. (2020)](https://arxiv.org/abs/2012.15781)), it is reasonable to freeze the majority of 
the parameters to save computation time, while still get sensible influence scores. 
Our interface also supports such operations.



Overall, under the hood, our "do-all" function `calculate_inflence_and_save` does the following:

```
for test_example in TestSet:
	inv_H = result from LiSSA  # extra sampling happenning here
	
	for train_example in TrainSet:
		score the train_example w.r.t. test_example, i.e. I(train_example, test_example)
	
	rank the scored TrainSet
  save test_example, top k train_examples, and bottom k train_examples
```



We want to note that improving the computation of the influence function remains an open question, 
so `calculate_inflence_and_save` is an expensive call. 
We hope to incorporate a speed-optimized interface proposed by [Guo et al. (2020)](https://arxiv.org/abs/2012.15781) in the future.

</exercise>