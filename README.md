# qM-AI-L

qM-AI-L is a project[^1] to evaluate and compare the performance of various pre-trained generative AI models in NLP on two distinct tasks:
* given an email body, generate a succinct subject line for it
* answer technical questions on AI / ML

## Email Subject Generation

qM-AI-L identifies the most salient words, phrases and sentences from the given email body and abstracts the message contained in that set into a very short, impactful subject line.

### Dataset

The pre-trained models are fine-tuned using the "[Annotated Enron Subject Line Corpus](https://github.com/ryanzhumich/AESLC)" dataset.
* The dataset consists of a subset of cleaned, filtered and deduplicated emails from the Enron Email Corpus which consists of employee email inboxes from the Enron Corporation.
* Evaluation (dev, test) split of the data contains 3 annotated subject lines by human annotators. Multiple possible references facilitate a better evaluation of the generated subject, since it is difficult to have only one unique, appropriate subject per email
* Some dataset statistics:
  * Sizes of train / dev / test splits: 14,436 / 1,960 / 1,906
  * An email contains an average of 75 words
  * A subject contains an average of 4 words

### Models

The models used are [t5-small](https://huggingface.co/google-t5/t5-small), [facebook/bart-base](https://huggingface.co/facebook/bart-base), [t5-base](https://huggingface.co/google-t5/t5-base), [unsloth/mistral-7b-bnb-4bit](https://huggingface.co/unsloth/mistral-7b-bnb-4bit)

Rouge-L, Meteor, Sacrebleu are some metrics used to compare the models.

| Model                  | Rouge-L  | Meteor | Sacrebleu|
|------------------------|----------|--------|----------|
| **t5-small**           |
| **t5-base**            |
| **facebook/bart-base** |
|**mistral-7b-bnb-4bit Unsloth**|0.5|0.4|0.3|
### Usage

The colab notebooks in this repo are self-contained and can be directly run.

## Question & Answer
TBD




[^1]: Capstone Project for PGCP program at IIITH by TalentSprint.
