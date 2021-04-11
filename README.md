# Unstructured Reviews Analysis

- The aim of this project is to extract meangingful insights from unstructured reviews about properties in different locations.
- These insights should be used later on for business needs!

## Dataset Brief Description

- The dataset contains mutliple reviews for different properties, each record in the dataset contain **user score** along with the actual **textual review**.
- Many reviews are in different languages **so language detection and translation is applied**.
- The dataset is **completely unlabelled**!

## Objectives

- Classify users' reviews into a **set of fixed topics**.
- Detect users' **sentiment** towards the property.
- Extract general insights about properties (property with highest +ve sentiment, topics frequency in each property, lowest rated property etc).

## Methods

1- Given a fixed set of topics, a huge list of synonyms have been extracted for each topic using **wordnet** and **word2vec**.

2- The extracted keywords for each topics has gone through _human filteriation stage to refine these keywords_

3- Applying **text mining and analysis** to label reviews to topics based on their **semantic and contextual meaning**

4- Using pretrained model for **sentiment analysis**

5- After doing the **unsupervised learning** on the reviews to label them into certain topics. Now our problem is a **supervised learning** one.

6- **Document Classification** Models should now be developed so any new documents could be classified! _(still in progress)_

## Sample Output for the unsupervised learning approach.

| Review  | Extracted Topics | Sentiment |
| ------- | ---------------- | --------- |
| "i just recently moved from millstead after being there for two in a half years. i can honestly say it was a pleasant experience. the staff was always nice erin jessica. also the maintenance crew did amazing fixing problems and keep the landscaping up. i moved from millstead because i wanted a change of scenery but i m definitely missing my old residence."  | `move`, `customer service`, `maintenance`, `landscaping` | Positive |
| "better maintenance team. so many half asses repairs done to my apartment and there is still a mosquito infestation in my place. walls are paper thin not repaired correctly painted over with the thickest coat imaginable making screws and vents hard to get to there's black mold literally everywhere in my ac ducts foundation problems galore and my carpets i have brought this up many times. i asked for it to be replaced because of mildew and shredded corners that are coming up. after many tries of getting it fixed we settled on just an additional cleaning. however it did not address the issue of the carpet coming up on all corners and being shredded from regular vacuuming." | `maintenance`, `customer service`, `apartment`, `pests`, `landscaping` | Negative

## Sample extracted insights from the reviews.

- Frequent words in reviews

![plot](https://github.com/OmarFarag95/insights-from-unstructured-reviews/blob/main/img/frequent_words.png)

- Topics frequency in all reviews

![plot](https://github.com/OmarFarag95/insights-from-unstructured-reviews/blob/main/img/topics_freqs.png)

- Statistics about **Terano** Property

![plot](https://github.com/OmarFarag95/insights-from-unstructured-reviews/blob/main/img/terano_stats.png)


- Statistics about **Berkshire** Property

![plot](https://github.com/OmarFarag95/insights-from-unstructured-reviews/blob/main/img/berkshire_stats.png)
