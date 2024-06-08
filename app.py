from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import os

app = Flask(__name__)

# Define the path to the model directory
model_directory = 'models/model'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)

# Load the model
model = AutoModelForSeq2SeqLM.from_pretrained(model_directory, local_files_only=True)

# Load the generation configuration
gen_config = GenerationConfig.from_pretrained(model_directory, local_files_only=True)

# Ensure necessary configuration entries are set
model.config.decoder_start_token_id = model.config.decoder_start_token_id if model.config.decoder_start_token_id is not None else model.config.bos_token_id
model.config.bos_token_id = model.config.bos_token_id if model.config.bos_token_id is not None else (tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0)
model.config.eos_token_id = model.config.eos_token_id if model.config.eos_token_id is not None else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2)
model.config.pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else tokenizer.pad_token_id

gen_config.max_length = gen_config.max_length if gen_config.max_length is not None else 142
gen_config.min_length = gen_config.min_length if gen_config.min_length is not None else 56
gen_config.early_stopping = gen_config.early_stopping if gen_config.early_stopping is not None else True
gen_config.num_beams = gen_config.num_beams if gen_config.num_beams is not None else 4
gen_config.length_penalty = gen_config.length_penalty if gen_config.length_penalty is not None else 2.0
gen_config.no_repeat_ngram_size = gen_config.no_repeat_ngram_size if gen_config.no_repeat_ngram_size is not None else 3
gen_config.forced_bos_token_id = gen_config.forced_bos_token_id if gen_config.forced_bos_token_id is not None else model.config.bos_token_id
gen_config.forced_eos_token_id = gen_config.forced_eos_token_id if gen_config.forced_eos_token_id is not None else model.config.eos_token_id

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}  # Ensure inputs are on the same device as the model
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=gen_config.max_length,
        min_length=gen_config.min_length,
        early_stopping=gen_config.early_stopping,
        num_beams=gen_config.num_beams,
        length_penalty=gen_config.length_penalty,
        no_repeat_ngram_size=gen_config.no_repeat_ngram_size,
        forced_bos_token_id=gen_config.forced_bos_token_id,
        forced_eos_token_id=gen_config.forced_eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,  # Explicitly pass decoder_start_token_id
        bos_token_id=model.config.bos_token_id  # Explicitly pass bos_token_id
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '')
    summary = summarize(text)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
