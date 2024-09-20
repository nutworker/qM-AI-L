***Question Answering on AIML Queries***

**Question and Answer Task**

-   The QnA task is to generate free-form responses that require not
    only finding relevant information from its training knowledge but
    also synthesizing this information into multiple accurate answer
    sentences.

**Model Approaches**

-   Both **extractive** and **sequence-to-sequence** approaches were
    explored for the given problem statement and fine-tuning. After
    careful evaluation, the Sequence-to-Sequence (Seq2Seq) approach was
    selected for the task due to following reasons:

    -   AI/ML knowledge corpus usage was not recommended to use

    -   Flexible Output Generation: Seq2Seq models generate new
        sequences, unlike extractive models, which are restricted to
        selecting text spans, making them ideal for tasks like
        summarization or translation.

**Dataset/ Data Preparation**

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

**Models' selection**

-   GPT-2 medium, Gemma 7B and Llama 3 8B were used for finetuning

  ------------------------------------------------------------------------
  **Feature**        **Gemma 7B**             **LLaMA 3 8B**
  ------------------ ------------------------ ----------------------------
  **Model Size**     7 billion parameters     8 billion parameters

  **Training Data**  Task-specific,           General-purpose, large
                     proprietary data         public datasets

  **Architecture**   Transformer-based,       Meta's LLaMA Transformer,
                     optimized for tasks      efficient design

  **Performance      Task/domain-specific     General-purpose NLP, high
  Focus**            fine-tuning              adaptability

  **Optimization**   Domain-specific          Lightweight, high
                     efficiency               generalization

  **Hardware         Lower resource           Slightly higher, still
  Requirements**     requirements             efficient

  **Fine-tuning      Focused on specific      Highly adaptable to various
  Flexibility**      tasks                    tasks

  **Use Case**       Specialized applications Broad range (generation,
                     (e.g., QA, support)      summarization, etc.)
  ------------------------------------------------------------------------

**Prompt Structure and Formatting**

**Fine-tuning Setup**

**Environment Setup**

-   Necessary libraries (e.g., PyTorch, Hugging Face Transformers) were
    set up

-   Necessary GPU/TPU resources were made available for handling large
    models efficiently.

**Model Loading**

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

**4. Fine-Tuning Configuration**

-   Hyperparameters are defined such as learning rate, batch size, and
    max sequence length.

**5. Training**

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

**Evaluation and performance**

-   Validate the model on a test set to check its performance and
    generalization ability.

-   Hyperparameters are adjusted and retrained until desired performance
    is reached.

> **Dataset-1:**

![](vertopal_3836775dcb2a420495fb59efac4ca3fe/media/image1.png){width="6.268055555555556in"
height="3.417361111111111in"}![](vertopal_3836775dcb2a420495fb59efac4ca3fe/media/image2.png){width="6.268055555555556in"
height="3.5833333333333335in"}

**Dataset-2**

![](vertopal_3836775dcb2a420495fb59efac4ca3fe/media/image3.png){width="6.268055555555556in"
height="3.3958333333333335in"}

![](vertopal_3836775dcb2a420495fb59efac4ca3fe/media/image4.png){width="6.268055555555556in"
height="3.917361111111111in"}

![](vertopal_3836775dcb2a420495fb59efac4ca3fe/media/image5.emf){width="6.268055555555556in"
height="1.7194444444444446in"}

  ------------------------------------------------------------------------
  **Dataset-2**                                  
  ----------------------------- ---------------- -------------------------
  **Metric**                    **Gemma_7b**     **Meta-Llama-3.1-8B**

  Avg_rouge1                    0.4389           0.4036

  Avg_rouge2                    0.2061           0.2035

  Avg_rougeL                    0.3709           0.3459

  Avg_rougeLsum                 0.3709           0.3459

  Avg_bleu_score                0.8368           0.4513

  Avg_meteor_score              0.2955           0.2853
  ------------------------------------------------------------------------

**Save and Deploy**

-   Fine-tuned model and tokenizer are saved for evaluation, inferencing
    and deployment

-   Deployed the model to production or use it for inference for
    generating answers
