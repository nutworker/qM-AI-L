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

### Methodology
* On high level different open source language models are researched and assesed that suits the problem statement of extracting most important words/ context/ concise summerisation. Transformer models and Bart models were found to be most apt for the given task other than the ChatGPT models.
* Couple of pretrained models were selected to test with zero-shot inferencing and further finetuning

  ### Models Performances

Rouge metrics were used to compare the models:

| Model                  | Rouge-1  |  Rouge-2  |  Rouge-L  |  Rouge-Lsum  | 
|------------------------|----------|-----------|-----------|--------------|
| **Flan T5 Base**       |0.3189      | 0.1852    |   0.3108   |    0.3100 |                 
| **facebook/bart-base** |0.2882      | 0.1232  |    0.2879   |    0.2893 
| **google/gemma-7b (unsloth)-ann1**    | 0.6355   |   0.4207    |    0.5924 |   0.5924 |
| **Mistral 7b (unsloth)**    | 0.2235   |  0.715    | 0.2236  | 0.2262 |
| **Phi-3 (unsloth)**    | 0.1063 | 0.0250 | 0.0942 | 0.0946 |*
 
  
### 1. Test the Pretrained Model with Zero Shot Inferencing
* Several models were loaded directly from hugging face and random records were inferenced to see how the models were behaving.
* **Google's Flan-T5, Facebook's Bart-Base, Gemma 7B**,  models were tried.
* By testing with various models with the zero shot inferencing, we can see that the model struggles to extract the same subject line compared to the human baseline subject, but it does pull out some important information from the email which indicates the models can be fine-tuned to the task at hand.

* ![image](https://github.com/user-attachments/assets/a9078e75-5aa7-4762-b7e7-d9ea8d14d0f1)

* Positional encoding:
 T5 uses relative position embeddings.
 BART uses absolute position embeddings.



### 2. Fine-Tune the Model with the Preprocessed Dataset
##### 2.1 - Preprocess the Email Dataset

Email-Subject (prompt-input-response) format is created as explicit instructions for the LLM. Prepend a prompt instruction to the start of email body and generate the subject with Suject as follows:

Training prompt (email):

prompt = f"""
Generate a subject line for the following email.

Email:
{email}

Subject:
"""

##### 2.2 - Fine-Tune the Model with the Preprocessed Dataset
* Utilize the built-in Hugging Face Trainer class. Pass the preprocessed dataset with reference to the original pretrained model. Several training parameters are tweeked and explored experimentally.
* Training a fully fine-tuned version of the model is taking few hours on a GPU. To save time, several checkpoints were created and the fully fine-tuned model were then initialised to use in the rest of experiments.
----------------------------------------------------------------------------------------
  
* ![image](https://github.com/user-attachments/assets/ea5d3021-68c5-4381-b1e8-ccf8b7fa50b2)

  ----------------------------------------------------------------------------------------

* ![image](https://github.com/user-attachments/assets/c748ffaf-f5bf-4946-9ed5-64d1a11ec89d)

------------------------------------------------------------------------------------------

![image](https://github.com/user-attachments/assets/1369382d-9382-46fb-b533-35c1040fac8a)

  

##### 2.3 - Evaluate the Model Qualitatively (Human Evaluation)
* Evaluated the model's performance qualitatively by comparing its ability to generate a reasonable subject line against its original subject to asses if the behaving the way it is supposed to, and is it able to understand the input. This approach confirmed that the fine-tuned model behaves as expected.


------------------------------------------------------------------------------------------
***Google's Flan-T5:**

   ![image](https://github.com/user-attachments/assets/acf4afc2-92fa-4066-a124-3a66aa80fc23)
  
------------------------------------------------------------------------------------------
  
***Facebook's Bart-Base:**

  ![image](https://github.com/user-attachments/assets/a142f1c1-530d-459e-90fb-b5e073ce6768)
  
------------------------------------------------------------------------------------------

 ***Gemma 7B:**
 
  ![image](https://github.com/user-attachments/assets/f757016f-aae5-42aa-8d1a-c544f0e8908a)

------------------------------------------------------------------------------------------

  
##### 2.4 - Evaluate the Model Quantitatively (with Rouge)

*The ROUGE metric helps quantify the validity of subject lines produced by models. It compares subjects to a "Annoted baseline" subject which is usually created by a human. While not perfect, it does indicate the overall increase in subject line generatiion effectiveness that we have accomplished by fine-tuning.

* Granularity: ROUGE-1 focuses on individual words, ROUGE-2 on word pairs, and ROUGE-L on the longest sequence of words.
Context: ROUGE-2 captures context better than ROUGE-1 due to its consideration of word pairs, while ROUGE-L and ROUGE-Lsum capture the overall sentence structure.
Summarization: ROUGE-Lsum is specifically designed for summarization, making it more relevant for evaluating the quality of summaries compared to ROUGE-L, which can be applied more generally.

Bleu measures precision: how much the words (and/or n-grams) in the machine generated summaries appeared in the human reference summaries.
Rouge measures recall: how much the words (and/or n-grams) in the human reference summaries appeared in the machine generated summaries.


* ![image](https://github.com/user-attachments/assets/e6f70672-7482-46f5-a802-4125b203dc49)

 -----------------------------------------------------------------------------------------

* ![image](https://github.com/user-attachments/assets/b02de9a9-6a4b-4cc2-8931-c6b23d0c0983)

 -----------------------------------------------------------------------------------------

* ![image](https://github.com/user-attachments/assets/6735dad2-625a-4ae1-8805-a6a7ec2a877e)
  
 -----------------------------------------------------------------------------------------

* notebook runtime got deleted ...crashed before saving/ commit

 -----------------------------------------------------------------------------------------

* ![image](https://github.com/user-attachments/assets/0c4ea92c-091c-4f79-a100-746fa6c16794)

   -----------------------------------------------------------------------------------------

*The Finetuning results showed improvement in all ROUGE metrics:

![image](https://github.com/user-attachments/assets/b486b11e-f522-4d3e-a082-e9b5a014bb55)


### Models Performances

Rouge metrics were used to compare the models:

| Model                  | Rouge-1  |  Rouge-2  |  Rouge-L  |  Rouge-Lsum  | 
|------------------------|----------|-----------|-----------|--------------|
| **Flan T5 Base**       |0.3189      | 0.1852    |   0.3108   |    0.3100 |                 
| **facebook/bart-base** |0.2882      | 0.1232  |    0.2879   |    0.2893 
| **google/gemma-7b (unsloth)-ann1**    | 0.6355   |   0.4207    |    0.5924 |   0.5924 |
| **Mistral 7b (unsloth)**    | 0.2235   |  0.715    | 0.2236  | 0.2262 |
| **Phi-3 (unsloth)**    | 0.1063 | 0.0250 | 0.0942 | 0.0946 |*

-----------------------------------------------------------------------------------------
*google/gemma-7b:
![image](https://github.com/user-attachments/assets/6c024dce-9035-4948-900c-e95015a3ee89)

-----------------------------------------------------------------------------------------

![image](https://github.com/user-attachments/assets/42e588f6-adbf-4a39-a286-9d2f7ae5a7e3)


## For FLAN T5 : Absolute percentage improvement of FINETUNED MODEL over PRETRAINED
rouge1: 6.82%
rouge2: 5.08%
rougeL: 7.08%
rougeLsum: 7.11%%

-----------------------------------------------------------------------------------------
### Observations:
* Fine-tuned models for generating email subject lines effectively capture key points and overall essence, with strong ROUGE-1 scores showing alignment with 
  essential topics.
* The model demonstrates potential for understanding nuanced details, as indicated by ROUGE-2 scores, though there is room for improvement.
* High ROUGE-L and ROUGE-Lsum scores reflect good maintenance of subject length and relevance.
* Specific prompts, such as "generate subject line," yield better results compared to combined prompts like "summarize and generate subject."
* Repetitive responses in pre-trained models (e.g., Mistral) were managed by applying a repetition_penalty of 1.5, but excessive penalties can cause unusual 
  outputs.
* Phi3 excels in text completion and GPT-style conversations but may produce hallucinations and less accurate results.

### 5 - Building App with Gradio and publishing in Hugging Face

# Build the Gradio App:#
We design our Gradio interface, defining how the user will interact with our model and ensuring the input and output specifications are clear.

# Save the App and Dependencies:
We prepare our app script and ensure all necessary dependencies are listed in a requirements file, ready for deployment.

# Publish on Hugging Face Spaces:
We create an account on Hugging Face, set up a new Space for our app, and push our code to this Space, making our app publicly accessible.



### Links to our project notebooks

https://github.com/nutworker/qM-AI-L/blob/main/Flan_T5_Base_Model.ipynb

https://github.com/nutworker/qM-AI-L/blob/L_test/email-subject/model-tuning/FB_Bart_Model1.ipynb

https://github.com/nutworker/qM-AI-L/blob/main/Gemma_7b_with_Unsloth2.ipynb

https://github.com/nutworker/qM-AI-L/blob/main/email-subject/model-tuning/Phi_3_Mini_4K_Instruct_Unsloth_2x_faster_finetuning_Group15.ipynb

https://github.com/nutworker/qM-AI-L/blob/main/email-subject/model-tuning/Alpaca_%2B_Mistral_v3_7b_full_example_Group15%20(1).ipynb


### Models References

The models used are [facebook/bart-base](https://huggingface.co/facebook/bart-base), [FLAN-T5](https://huggingface.co/docs/transformers/en/model_doc/flan-t5#overview) [phi3](https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing) [Gemma-7b](https://huggingface.co/unsloth/gemma-7b-bnb-4bit) [Mistral](https://colab.research.google.com/drive/1_yNCks4BTD5zOnjozppphh5GzMFaMKq_?usp=sharing) [google-gemma-with-unsloth](https://www.analyticsvidhya.com/blog/2024/04/fine-tuning-google-gemma-with-unsloth/)

### Usage

The colab notebooks in this repo are self-contained and can be directly run.


### Next Steps : Perform Parameter Efficient Fine-Tuning (PEFT)

* Parameter Efficient Fine-Tuning (PEFT), which is more efficient than full fine-tuning and yields comparable results. PEFT, often referring to Low-Rank Adaptation (LoRA), enables fine-tuning with fewer compute resources, often a single GPU. Tried PEFT on Flan T5 Base Model

[^1]: Capstone Project for PGCP program at IIITH by TalentSprint.
