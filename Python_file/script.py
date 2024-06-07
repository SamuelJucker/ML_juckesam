import os

# Define the path to the model directory
model_directory = 'model'

# List of required files
required_files = [
    'config.json',
    'generation_config.json',
    'merges.txt',
    'model.safetensors',  # or 'pytorch_model.bin', depending on your model
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.json'
]

# Check if all required files are present
missing_files = [file for file in required_files if not os.path.isfile(os.path.join(model_directory, file))]

if missing_files:
    print(f"Missing files: {missing_files}")
else:
    print("All required files are present.")

# If all files are present, proceed to load the model and tokenizer
if not missing_files:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory, local_files_only=True)

    # Load the model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_directory, local_files_only=True)

    # Load the generation configuration
    gen_config = GenerationConfig.from_pretrained(model_directory, local_files_only=True)

    #
