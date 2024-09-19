
Question Answering on AIML Queries

Question and Answer Task
•	The QnA task is to generate free-form responses that require not only finding relevant information from its training knowledge but also synthesizing this information into multiple accurate answer sentences.
Model Approaches
•	Both extractive and sequence-to-sequence approaches were explored for the given problem statement and fine-tuning. After careful evaluation, the Sequence-to-Sequence (Seq2Seq) approach was selected for the task due to following reasons:
o	AI/ML knowledge corpus usage was not recommended to use 
o	Flexible Output Generation: Seq2Seq models generate new sequences, unlike extractive models, which are restricted to selecting text spans, making them ideal for tasks like summarization or translation.
Dataset/ Data Preparation
•	Input-Output Pairs: For Seq2Seq fine-tuning the model is provided with structured input-output pairs, where:
o	Input: The question without a context
o	Output: The target answer.
•	A total of 462 question-answer pairs were collaboratively prepared from the AIML course. The dataset was coalited in the prescribed format in the CSV file.
•	A consolidated train/dev/test set was provided for further fine-tuning with the GPT variant model.
•	Dataset-1 has a question-and-answer pair for train set and, a question and two human annotated answers for the test and dev sets.
o	Train set -(1316, 2)
o	Test set (120, 3)
o	Dev set (80, 3)
•	Dataset-2 has a question-and-answer pair
o	Train set -(1985, 2)
o	Test set -(249, 2)
o	Dev set - (248, 2))

Models’ selection
•	GPT-2 medium, Gemma 7B and Llama 3 8B were used for finetuning

Feature	Gemma 7B	LLaMA 3 8B
Model Size	7 billion parameters	8 billion parameters
Training Data	Task-specific, proprietary data	General-purpose, large public datasets
Architecture	Transformer-based, optimized for tasks	Meta’s LLaMA Transformer, efficient design
Performance Focus	Task/domain-specific fine-tuning	General-purpose NLP, high adaptability
Optimization	Domain-specific efficiency	Lightweight, high generalization
Hardware Requirements	Lower resource requirements	Slightly higher, still efficient
Fine-tuning Flexibility	Focused on specific tasks	Highly adaptable to various tasks
Use Case	Specialized applications (e.g., QA, support)	Broad range (generation, summarization, etc.)

Prompt Structure and Formatting

Fine-tuning Setup

Environment Setup
•	Necessary libraries (e.g., PyTorch, Hugging Face Transformers) were set up
•	Necessary GPU/TPU resources were made available for handling large models efficiently.
Model Loading
•	Load the pretrained model (Gemma 7B or LLaMA 8B) from Huggingface using FastLanguageModel/ SFTTrainer class
•	Create a PEFT model with the given parameters and load adapters - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning with following parameters
o	    r=16, # LoRa Rank
o	    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
o	                      "gate_proj", "up_proj", "down_proj",],
o	    lora_alpha=16,
o	    lora_dropout=0,
o	    bias="none",
o	    use_gradient_checkpointing=True
4. Fine-Tuning Configuration
•	Hyperparameters are defined such as learning rate, batch size, and max sequence length.
5. Training
•	Perform backpropagation to adjust model weights based on the task-specific loss function.
•	Monitor model performance (validation loss, accuracy) during training to prevent overfitting.
•	TrainingArguments used as below
o	        per_device_train_batch_size = 1,
o	        gradient_accumulation_steps = 2,
o	        warmup_steps = 5,
o	        max_steps = 30,
o	        learning_rate = 2e-4,
o	        fp16 = not torch.cuda.is_bf16_supported(),
o	        bf16 = torch.cuda.is_bf16_supported(),
o	        logging_steps = 1,
o	        optim = "paged_adamw_8bit",
o	        weight_decay = 0.01,
o	        lr_scheduler_type = "linear",
o	        seed = 3407,
o	        output_dir = "outputs",
Evaluation and performance
•	Validate the model on a test set to check its performance and generalization ability.
•	Hyperparameters are adjusted and retrained until desired performance is reached.
Dataset-1:
  

Dataset-2
 

 
 

Dataset-2
Metric	Gemma_7b	Meta-Llama-3.1-8B
Avg_rouge1	0.4389	0.4036
Avg_rouge2	0.2061	0.2035
Avg_rougeL	0.3709	0.3459
Avg_rougeLsum	0.3709	0.3459
Avg_bleu_score	0.8368	0.4513
Avg_meteor_score	0.2955	0.2853

Save and Deploy
•	Fine-tuned model and tokenizer are saved for evaluation, inferencing and deployment
•	Deployed the model to production or use it for inference for generating answers
