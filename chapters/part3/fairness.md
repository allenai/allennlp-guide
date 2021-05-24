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
This section only provides a high-level overview of fairness and how NLP models can <em>encode</em> and <em>amplify</em> biases. We highly encourage readers to consult the works of leading scholars (particularly Black women) to dive further into this field. Some excellent papers <em>in the NLP domain</em> with which to begin your journey include:

1. Blodgett, S.L., Barocas, S., Daumé, H., & Wallach, H. (2020). Language (Technology) is Power: [A Critical Survey of "Bias" in NLP](https://api.semanticscholar.org/CorpusID:218971825). ACL.

2. Chang, K.W., Ordonez, V., Mitchell, M., Prabhakaran, V (2019). [Tutorial: Bias and Fairness in Natural Language Processing](http://web.cs.ucla.edu/~kwchang/talks/emnlp19-fairnlp/). EMNLP 2019.

3. Barocas, S., Hardt, M., and Narayanan, A. 2019. [Fairness and machine learning](https://fairmlbook.org).

4. Bender, E.M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big? 🦜](https://api.semanticscholar.org/CorpusID:232040593). Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency.

In the rest of this section, we motivate the need for fairness and bias mitigation tools in NLP, based on References 1, 2 and 3 above.

<em>Fairness</em> concerns ensuring that a model performs equitably for all groups of people with respect to protected attributes. For instance, we may want a model to correctly predict `[MASK]` tokens at the same rate for men, women, and non-binary genders.

<em>Biases</em> are stereotypical/unjust associations a model encodes with respect to protected attributes, or subpar model performance for certain groups. Hence, <em>bias mitigation</em> is the process of lessening the severity of stereotypical/unjust associations and disparate model performance. For example, we may want to disassociate all occupations from gender, to mitigate gender bias in a downstream application that recommends jobs to individuals. 

## Biases in Data

We are all aware that large language models are hungry for text data, and these data greatly influence the word and sentence-level representations the models learn. NLP researchers and pratictioners often scrape data from the Internet, which include news articles, Wikipedia pages, and even patents! But, how representative are the text in these data of the real-world frequences of different words or groups of people? The answer is **not at all**.
<br>
<br>
This is the result of <em>reporting bias</em>, in which the "frequency with which people write about actions, outcomes, or properties is not a reflection of real-world frequencies or the degree to which a property is characteristic of a class of individuals" [2]. For instance, "murdered" occurs with a much higher frequency than does "blinked" in text corpuses because journalists usually report about murders rather than blinking, ironically as a result of the relative rarity of murders [2].

There are many other biases in data that affect language models [2]:
1. **Selection bias:** Data selection does not reflect a random sample, e.g. English language data are not representative of every dialect of English, Wikipedia pages overrepresent men 

2. **Out-group homogeneity bias:** People tend to see outgroup members as more alike than ingroup members when comparing attitudes, values, personality traits, and other characteristics, e.g. researchers incorrectly treat "non-binary" as a single, cohesive gender

3. **Biased data representation:** Even with an appropriate amount of data for every group, some groups are represented less positively than others, e.g. Wikipedia pages exhibit serious biases against women

4. **Biased labels:** Annotations in dataset reflect the worldviews of annotators, e.g. annotators prescribe incorrect labels due to an unawareness of non-Western cultures and traditions

Barocas and Selbst, in their 2016 paper "[Big Data's Disparate Impact](https://api.semanticscholar.org/CorpusID:143133374)," also broke down biases in data into the following categories (which overlap with the categories above) [3]:

1. **Skewed sampling (feedback loop):** Future observations confirm a model's predictions, which leads to fewer opportunities to make observations that contradict the model's predictions, thereby causing any biases to compound, e.g. predictive policing tools are influenced by their own predictions of where crime will occur

2. **Tainted examples:** Social biases like racism, sexism, and homophobia taint datasets, and then get baked into models which are trained and evaluated on these datasets, e.g. Amazon's automated resume screening system identified women to be unfit for engineering roles because of the company's poor track record of hiring women

3. **Limited features:** Features may be less informative or less reliably collected for certain groups of people, which leads to a model exhibiting disparate errors for different groups

4. **Sample size disparities:** There exist disparities in the amount of data across certain groups of people, which leads to a model exhibiting disparate errors for different groups

5. **Proxies:** Even after the removal of a feature which a model should not consider (e.g. race), there exist other features with a high correlation with the undesirable feature that the model can (unfairly) leverage (e.g. zip code)

Please note that every dataset is biased, and it is <em>impossible</em> to fully remove biases from datasets; even though it may be relatively easy to identify and remove explicit social biases, it is difficult to eliminate skewed samples and proxies that could reinforce these biases. That being said, it's still a great idea to create [datasheets for your datasets](https://api.semanticscholar.org/CorpusID:4421027).

## Biases in Models

You may often hear, in the context of biases, "It's not the model, it's the data!" This is far from the truth. Models are excellent at <em>encoding</em> and <em>amplifying</em> biases. 

1. **Imbalanced datasets:** Model performance inherently favors groups with rich features and a large, diverse sample, thereby marginalizing groups with limited features and a poor sample size. 

2. **Spurious correlations:** Models learn complex, non-linear and spurious correlations between features and annotations that often reflect social biases and are difficult to detect and eliminate. For example, [Wang et al.](https://api.semanticscholar.org/CorpusID:195847929) discovered that even after removing subjects from images, a model could still easily predict the gender of subjects because women were photographed overwhelmingly frequently in the kitchen, that is, the model leveraged the existence of a kitchen in the image as a proxy for gender.

3. **Contextualization:** LSTMs and attention-based models, through contextualization, allow for biases to be propagated across words in a sentence.

Balanced datasets are not effective at taming bias amplification because of latent variables.

## Biases in Interpretation

Beyond biases in data and models, there also exist biases in interpretation [2]:

1. **Confirmation bias:** The tendency to search for, interpret, favor, recall information in a way that confirms preexisting beliefs, e.g. researchers misconstrue the ability of a model to predict binary gender as the non-existence of non-binary genders

2. **Overgeneralization:** Coming to conclusions based on information that is too general, e.g. researchers conclude that a model works well for everyone based on its overall accuracy on data containing mostly white people

3. **Correlation fallacy:** Confusing correlation with causation

4. **Automation bias:** Propensity for humans to favor suggestions from automated decision-making systems over contradictory information without automation

Biases in interpretation truly necessitate the creation and usage of metrics that can uncover unfairness and biased associations in models. 

## Call to Action

Why should we care about fairness and biases? Because fairness and mitigating biases are inherently normative [1]. As language models become more prevalent in critical decision-making systems, their unfair encoding and amplification of biases can pose serious harms for already-marginalized communities. In NLP literature, harms are often categorized into the following (Barocas and Selbst, 2016):

1. **Representational harms:** when systems reinforce the subordination of some groups along the lines of identity
2. **Allocational harms:** when a system allocates or withholds a certain opportunity or resource

This is one of many reasons to use [model cards for model reporting](https://dl.acm.org/doi/10.1145/3287560.3287596).

Can biases be removed? No. However, they can be reduced and controlled. Completely removing biases is difficult, especially with large language models, because of contextualization, spurious correlations, and latent variables.

</exercise>

<exercise id="2" title="An Overview of Fairness Metrics">
</exercise>

<exercise id="3" title="An Overview of Bias Mitigators">
</exercise>

<exercise id="4" title="An Overview of Bias Metrics">
</exercise>

<exercise id="5" title="Applying Bias Mitigation to Large Transformers">
</exercise>

<exercise id="6" title="Next Steps">
We'd love contributions to the Fairness module!
</exercise>
