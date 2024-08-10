import gradio as gr
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Replace with your model name
#MODEL_NAME = "ssirikon/Gemma7b-bnb-Unsloth"
#MODEL_NAME = "unsloth/gemma-7b-bnb-4bit"
MODEL_NAME = "Lohith9459/gemma7b"

# Load the model and tokenizer
max_seq_length = 512
dtype = torch.bfloat16
load_in_4bit = True

#model = FastLanguageModel.from_pretrained(MODEL_NAME, max_seq_length=max_seq_length, dtype=dtype, load_in_4bit=load_in_4bit)
#tokenizer = model.tokenizer

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_subject(email_body):
  instruction = "Generate a subject line for the following email."
  formatted_text = f"""Below is an instruction that describes a task. \
    Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {email_body}
    ### Response:
    """
  inputs = tokenizer([formatted_text], return_tensors="pt").to("cuda")
  text_streamer = TextStreamer(tokenizer)
  generated_ids = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)
  generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

  def extract_subject(text):
    start_tag = "### Response:"
    start_idx = text.find(start_tag)
    if start_idx == -1:
        return None
    subject = text[start_idx + len(start_tag):].strip()
    return subject

  return extract_subject(generated_text)

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_subject,
    inputs=gr.Textbox(lines=20, label="Email Body"),
    outputs=gr.Textbox(label="Generated Subject")
)

demo.launch()