# qM-AI-L

qM-AI-L is a project[^1] to evaluate and compare the performance of various pre-trained generative AI models in NLP on two distinct tasks:
* given an email body, generate a succinct subject line for it
* answer technical questions on AI / ML

## Email Subject Generation

qM-AI-L identifies the most salient words, phrases and sentences from the given email body and abstracts the message contained in that set into a very short, impactful subject line.

### Dataset

The pre-trained models are fine-tuned using the "[Annotated Enron Subject Line Corpus](https://github.com/ryanzhumich/AESLC)" dataset.
* The dataset consists of a subset of cleaned, filtered and deduplicated emails from the Enron Email Corpus which consists of employee email inboxes from the Enron Corporation.

#### Data Loading and Pre-Processing
* LangChain_community.DirectoryLoaders are used to load the email files and then converted to Pandas DataFrame.(LangChain document_loader was found to be organized, scalable, easy to use)
* Evaluation (dev, test) split of the data contains 3 annotated subject lines by human annotators. Multiple possible references facilitate a better evaluation of the generated subject, since it is difficult to have only one unique, appropriate subject per email
* Some dataset statistics:
  * Sizes of train / dev / test splits: 14,436 / 1,960 / 1,906
  * An email contains an average of 75 words
  * A subject contains an average of 4 words
 * A subset of train dataset is created for finetuning language models, although full train data set is also used a couple of times. 

#### Methodology
* On high level different open source language models are researched and assesed that suits the problem statement of extracting most important words and concise summerisation. Transformer models and Bart models were found to be most apt for the given task other than the ChatGPT models.
* Couple of pretrained models were selected to test with zero-shot inferencing and further finetuning
  
##### 1. Test the Pretrained Model with Zero Shot Inferencing
* Several models were loaded directly from hugging face and random records were inferenced to see how the models were behaving.
* Google's T5-small, T5-Base, Flan-T5 and Facebook's Bart-Base models were tried.
* By testing with various models with the zero shot inferencing, we can see that the model struggles to extract the same subject line compared to the human baseline subject, but it does pull out some important information from the email which indicates the models can be fine-tuned to the task at hand.
* 
##### 2. Fine-Tune the Model with the Preprocessed Dataset
###### 2.1 - Preprocess the Email Dataset

Email-Subject (prompt-input-response) format is created as explicit instructions for the LLM. Prepend a prompt instruction to the start of email body and generate the subject with Suject as follows:

Training prompt (email):

prompt = f"""
Generate the subject line for the following email.

Email:
{email}

Subject:
"""

###### 2.2 - Fine-Tune the Model with the Preprocessed Dataset
* Utilize the built-in Hugging Face Trainer class. Pass the preprocessed dataset with reference to the original pretrained model. Several training parameters are tweeked and explored experimentally.
* Training a fully fine-tuned version of the model is taking few hours on a GPU. To save time, several checkpoints were created and the fully fine-tuned model were then initialised to use in the rest of experiments.
* ![image](https://github.com/user-attachments/assets/ea5d3021-68c5-4381-b1e8-ccf8b7fa50b2)

* 

###### 2.3 - Evaluate the Model Qualitatively (Human Evaluation)
* Evaluated the model's performance qualitatively by comparing its ability to generate a reasonable subject line against its original subject to asses if the behaving the way it is supposed to, and is it able to understand the input. This approach confirmed that the fine-tuned model behaves as expected.
  
* ![image](https://github.com/user-attachments/assets/acf4afc2-92fa-4066-a124-3a66aa80fc23)

###### 2.4 - Evaluate the Model Quantitatively (with ROUGE Metric etc)

*The ROUGE metric helps quantify the validity of subject lines produced by models. It compares subjects to a "Annoted baseline" subject which is usually created by a human. While not perfect, it does indicate the overall increase in subject line generatiion effectiveness that we have accomplished by fine-tuning.

* 

* ![image](https://github.com/user-attachments/assets/b02de9a9-6a4b-4cc2-8931-c6b23d0c0983)

* ![image](https://github.com/user-attachments/assets/6735dad2-625a-4ae1-8805-a6a7ec2a877e)
  
* notebook crashed before saving/ commit

* ![image](https://github.com/user-attachments/assets/0c4ea92c-091c-4f79-a100-746fa6c16794)



##### 3 - Perform Parameter Efficient Fine-Tuning (PEFT)
3.1 - Setup the PEFT/LoRA model for Fine-Tuning
3.2 - Train PEFT Adapter
3.3 - Evaluate the Model Qualitatively (Human Evaluation)
3.4 - Evaluate the Model Quantitatively (with ROUGE Metric)

#### Models

The models used are [t5-small](https://huggingface.co/google-t5/t5-small), [facebook/bart-base](https://huggingface.co/facebook/bart-base), [t5-base](https://huggingface.co/google-t5/t5-base).

Rouge-L, Meteor, Sacrebleu are some metrics used to compare the models.

| Model                  | Rouge-L  | Meteor | Sacrebleu|
|------------------------|----------|--------|----------|
| **t5-small**           |
| **t5-base**            |
| **facebook/bart-base** |

### Usage

The colab notebooks in this repo are self-contained and can be directly run.

## Question & Answer
TBD




[^1]: Capstone Project for PGCP program at IIITH by TalentSprint.
