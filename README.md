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
  
#### 1. Test the Pretrained Model with Zero Shot Inferencing
* Several models were loaded directly from hugging face and random records were inferenced to see how the models were behaving.
* Google's T5-small, T5-Base, Flan-T5 and Facebook's Bart-Base models were tried.
* By testing with various models with the zero shot inferencing, we can see that the model struggles to extract the same subject line compared to the human baseline subject, but it does pull out some important information from the email which indicates the models can be fine-tuned to the task at hand.

* ![image](https://github.com/user-attachments/assets/a9078e75-5aa7-4762-b7e7-d9ea8d14d0f1)


#### 2. Fine-Tune the Model with the Preprocessed Dataset
##### 2.1 - Preprocess the Email Dataset

Email-Subject (prompt-input-response) format is created as explicit instructions for the LLM. Prepend a prompt instruction to the start of email body and generate the subject with Suject as follows:

Training prompt (email):

prompt = f"""
Generate the subject line for the following email.

Email:
{email}

Subject:
"""

##### 2.2 - Fine-Tune the Model with the Preprocessed Dataset
* Utilize the built-in Hugging Face Trainer class. Pass the preprocessed dataset with reference to the original pretrained model. Several training parameters are tweeked and explored experimentally.
* Training a fully fine-tuned version of the model is taking few hours on a GPU. To save time, several checkpoints were created and the fully fine-tuned model were then initialised to use in the rest of experiments.
  
* ![image](https://github.com/user-attachments/assets/ea5d3021-68c5-4381-b1e8-ccf8b7fa50b2)

* ![image](https://github.com/user-attachments/assets/c748ffaf-f5bf-4946-9ed5-64d1a11ec89d)


  

##### 2.3 - Evaluate the Model Qualitatively (Human Evaluation)
* Evaluated the model's performance qualitatively by comparing its ability to generate a reasonable subject line against its original subject to asses if the behaving the way it is supposed to, and is it able to understand the input. This approach confirmed that the fine-tuned model behaves as expected.
  
* ![image](https://github.com/user-attachments/assets/acf4afc2-92fa-4066-a124-3a66aa80fc23)

##### 2.4 - Evaluate the Model Quantitatively (with ROUGE Metric etc)

*The ROUGE metric helps quantify the validity of subject lines produced by models. It compares subjects to a "Annoted baseline" subject which is usually created by a human. While not perfect, it does indicate the overall increase in subject line generatiion effectiveness that we have accomplished by fine-tuning.

* Granularity: ROUGE-1 focuses on individual words, ROUGE-2 on word pairs, and ROUGE-L on the longest sequence of words.
Context: ROUGE-2 captures context better than ROUGE-1 due to its consideration of word pairs, while ROUGE-L and ROUGE-Lsum capture the overall sentence structure.
Summarization: ROUGE-Lsum is specifically designed for summarization, making it more relevant for evaluating the quality of summaries compared to ROUGE-L, which can be applied more generally.

* ![image](https://github.com/user-attachments/assets/e6f70672-7482-46f5-a802-4125b203dc49)

* ![image](https://github.com/user-attachments/assets/b02de9a9-6a4b-4cc2-8931-c6b23d0c0983)

* ![image](https://github.com/user-attachments/assets/6735dad2-625a-4ae1-8805-a6a7ec2a877e)
  
* notebook crashed before saving/ commit

* ![image](https://github.com/user-attachments/assets/0c4ea92c-091c-4f79-a100-746fa6c16794)

*The Finetuning results showed improvement in all ROUGE metrics:

![image](https://github.com/user-attachments/assets/b486b11e-f522-4d3e-a082-e9b5a014bb55)



##### 3 - Perform Parameter Efficient Fine-Tuning (PEFT)

* Parameter Efficient Fine-Tuning (PEFT), which is more efficient than full fine-tuning and yields comparable results. PEFT, often referring to Low-Rank Adaptation (LoRA), enables fine-tuning with fewer compute resources, often a single GPU. Tried PEFT on Flan T5 Base Model

##### 3.1 - Setup the PEFT/LoRA model for Fine-Tuning

* LoRA produces a small adapter (a few MBs) while keeping the original LLM unchanged. During inference, this adapter is combined with the original LLM, allowing multiple adapters to reuse the same LLM and reducing memory requirements for various tasks.

* Total Trainable parameters have decreased considerably.
          
    peft_model = get_peft_model(flan_t5_base_original_model, lora_config)
    print(print_number_of_trainable_model_parameters(peft_model))
    Trainable model parameters: 3538944
    All model parameters: 251073792
    Percentage of trainable model parameters: 1.41%
  
##### 3.2 - Train PEFT Adapter

* loaded the model with lora_config
* ![image](https://github.com/user-attachments/assets/6d58280f-5e43-4185-a812-469da50ea51f)
 
* Tried highr learning_rate=1e-4, # Higher learning rate than full fine-tuning.
* ![image](https://github.com/user-attachments/assets/6e392bcd-e1fa-4576-8eeb-d28e8255237c)


##### 3.3 - Evaluate the Model Qualitatively (Human Evaluation)

##### 3.4 - Evaluate the Model Quantitatively (with ROUGE Metric)

### Models References

The models used are [t5-small](https://huggingface.co/google-t5/t5-small), [facebook/bart-base](https://huggingface.co/facebook/bart-base), [t5-base](https://huggingface.co/google-t5/t5-base).

[FLAN-T5](https://huggingface.co/docs/transformers/en/model_doc/flan-t5#overview)

Rouge-L, Meteor, Sacrebleu are some metrics used to compare the models.

| Model                  | Rouge-L  | Meteor | Sacrebleu|
|------------------------|----------|--------|----------|
| **t5-small**           |
| **t5-base**            |
| **Flan T5 Base**
| **facebook/bart-base** |

### Usage

The colab notebooks in this repo are self-contained and can be directly run.

## Question & Answer
TBD




[^1]: Capstone Project for PGCP program at IIITH by TalentSprint.
