# Question Answering on AIML Queries
---
## Task

-   The QnA task is to generate free-form responses that require not
    only finding relevant information from its training knowledge but
    also synthesizing this information into multiple accurate answer
    sentences.
---
## Model Approaches

-   Both **extractive** and **sequence-to-sequence** approaches were
    explored for the given problem statement and fine-tuning. After
    careful evaluation, the Sequence-to-Sequence (Seq2Seq) approach was
    selected for the task due to following reasons:

    -   AI/ML knowledge corpus usage was not recommended to use

    -   Flexible Output Generation: Seq2Seq models generate new
        sequences, unlike extractive models, which are restricted to
        selecting text spans, making them ideal for tasks like
        summarization or translation.
---
## Dataset/ Data Preparation

-   Input-Output Pairs: For Seq2Seq fine-tuning the model is provided
    with structured input-output pairs, where:

    -   Input: The question without a context

    -   Output: The target answer.

-   A total of 462 question-answer pairs were collaboratively prepared
    from the AIML course. The dataset was coalited in the prescribed
    format in the CSV file.

-   A consolidated train/dev/test set was provided for further
    fine-tuning with the GPT variant model.

-   Dataset-1 has a question-and-answer pair for train set and, a
    question and two human annotated answers for the test and dev sets.

    -   Train set -(1316, 2)

    -   Test set (120, 3)

    -   Dev set (80, 3)

-   Dataset-2 has a question-and-answer pair

    -   Train set -(1985, 2)

    -   Test set -(249, 2)

    -   Dev set - (248, 2))

---
## Models' selection

-   GPT-2 medium, Gemma 7B and Llama 3 8B were used for finetuning


![image](https://github.com/user-attachments/assets/24c77c1f-8207-4d47-8ea5-81018168b04d)


 ------------------------------------------------------------------------
# Fine-tuning

## Environment Setup

-   Necessary libraries (e.g., PyTorch, Hugging Face Transformers) were
    set up

-   Necessary GPU/TPU resources were made available for handling large
    models efficiently.
--
##  Model Loading

-   Load the pretrained model (Gemma 7B or LLaMA 8B) from Huggingface
    using FastLanguageModel/ SFTTrainer class

-   Create a PEFT model with the given parameters and load adapters -
    LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning with
    following parameters

    -   r=16, \# LoRa Rank

    -   target_modules=\[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\",

    -   \"gate_proj\", \"up_proj\", \"down_proj\",\],

    -   lora_alpha=16,

    -   lora_dropout=0,

    -   bias=\"none\",

    -   use_gradient_checkpointing=True

  ------------------------------------------------------------------------

## Prompt Structure and Formatting

* Added Clear Instructions: Specified format, tone, or length to guide the model’s response.
* Used Examples: Provided sample responses to show the desired structure and style.

* GPT2LMHead Model doesn’t need a context to be provided to generate a response unlike GPT2ForQuestionAnswering.
* Compared to GPT2 advanced models like gemma provide better answers as they have been trained on lot of data.
* Prompt given makes a difference in the predicted response.

* Llama model was generating answers with http links/ references from its earlier trained knowledge. Solved it with giving prompt instruction.
* TextStreamer was not respecting EOS_TOKEN for few questions. Continuous answer generation. Debug EOS Token Behavior needs to done.




------------------------------------------------------------------------
## Fine-Tuning Configuration

-   Hyperparameters are defined such as learning rate, batch size, and
    max sequence length.

## Training

-   Perform backpropagation to adjust model weights based on the
    task-specific loss function.

-   Monitor model performance (validation loss, accuracy) during
    training to prevent overfitting.

-   TrainingArguments used as below

    -   per_device_train_batch_size = 1,

    -   gradient_accumulation_steps = 2,

    -   warmup_steps = 5,

    -   max_steps = 30,

    -   learning_rate = 2e-4,

    -   fp16 = not torch.cuda.is_bf16_supported(),

    -   bf16 = torch.cuda.is_bf16_supported(),

    -   logging_steps = 1,

    -   optim = \"paged_adamw_8bit\",

    -   weight_decay = 0.01,

    -   lr_scheduler_type = \"linear\",

    -   seed = 3407,

    -   output_dir = \"outputs\",
 
![image](https://github.com/user-attachments/assets/fd8af85d-2a0e-4d21-a1f2-1be6a6c85117)

---
## Evaluation and performance

-   Validate the model on a test set to check its performance and
    generalization ability.

-   Hyperparameters are adjusted and retrained until desired performance
    is reached.

# Dataset-1:
![image](https://github.com/user-attachments/assets/16750d81-96d0-4e50-add1-a2a3f2ddb1ba)
--
![image](https://github.com/user-attachments/assets/abff7f58-a234-4978-a510-ba606d8f4ef1)
--

--
# Dataset-2

![image](https://github.com/user-attachments/assets/a52724e5-2d07-4f7f-9f90-210604ce2d7b)
--
![image](https://github.com/user-attachments/assets/5692a685-ce7b-4062-855a-d6855508f1ca)


------------------------------------------------------------------------
![image](https://github.com/user-attachments/assets/d88783c7-c215-4fb6-834c-9131cca8431c)



![image](https://github.com/user-attachments/assets/be0a952c-03ef-4e64-9a6b-6ad99558e171)



  ------------------------------------------------------------------------

## Save and Deploy

Build the Gradio App: Designed Gradio interface, defining how the user will interact with the model and ensuring the input and output specifications are clear.

Save the App and Dependencies: Prepared our app script and ensure all necessary dependencies are listed in a requirements file, ready for deployment.

Publish on Hugging Face Spaces: Created an account on Hugging Face, set up a new Space for our app, and push our code to this Space, making our app publicly accessible.


