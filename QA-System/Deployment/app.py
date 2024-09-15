import gradio as gr
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your model name
#MODEL_NAME = "ssirikon/Gemma7b-bnb-Unsloth"
#MODEL_NAME = "unsloth/gemma-7b-bnb-4bit"
MODEL_NAME = "Lohith9459/QnAD2_gemma7b"

# Load the model and tokenizer
max_seq_length = 512
dtype = torch.bfloat16
load_in_4bit = True

#model = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=load_in_4bit)
#tokenizer = model.tokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_answer(question):
  instruction = "Generate an answer for the following question in less than two sentences."
  formatted_text = f"""Below is an instruction that describes a task. \
    Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {question}
    ### Response:
    """
  inputs = tokenizer([formatted_text], return_tensors="pt").to("cuda")
  text_streamer = TextStreamer(tokenizer)
  generated_ids = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)
  generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

  def get_answer(text):
      start_tag = "### Response:"

      # Find the start and end indices
      start_idx = text.find(start_tag)

      # Check if both tags are found
      if start_idx == -1:
          return None  # Tags not found

      # Extract content between the tags
      answer = text[start_idx + len(start_tag):].strip()

      return answer

  return get_answer(generated_text)


# Create the Gradio interface
demo = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(lines=5, label="Ask Question on AI/ML"),
    outputs=gr.Textbox(label="G-15 Gemma7b Model Generated Answer")
)

demo.launch()