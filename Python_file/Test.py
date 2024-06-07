from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import os

# Define the path to the model directory
model_directory = 'model'

# Check for the presence of required files
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

missing_files = [file for file in required_files if not os.path.isfile(os.path.join(model_directory, file))]

if missing_files:
    print(f"Missing files: {missing_files}")
else:
    print("All required files are present.")

# If all files are present, proceed to load the model and tokenizer
if not missing_files:
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

    # Print out the config values to verify they are set correctly
    print(f"decoder_start_token_id: {model.config.decoder_start_token_id}")
    print(f"bos_token_id: {model.config.bos_token_id}")
    print(f"eos_token_id: {model.config.eos_token_id}")
    print(f"pad_token_id: {model.config.pad_token_id}")

    # Ensure generation config entries are set
    gen_config.max_length = gen_config.max_length if gen_config.max_length is not None else 142
    gen_config.min_length = gen_config.min_length if gen_config.min_length is not None else 56
    gen_config.early_stopping = gen_config.early_stopping if gen_config.early_stopping is not None else True
    gen_config.num_beams = gen_config.num_beams if gen_config.num_beams is not None else 4
    gen_config.length_penalty = gen_config.length_penalty if gen_config.length_penalty is not None else 2.0
    gen_config.no_repeat_ngram_size = gen_config.no_repeat_ngram_size if gen_config.no_repeat_ngram_size is not None else 3
    gen_config.forced_bos_token_id = gen_config.forced_bos_token_id if gen_config.forced_bos_token_id is not None else model.config.bos_token_id
    gen_config.forced_eos_token_id = gen_config.forced_eos_token_id if gen_config.forced_eos_token_id is not None else model.config.eos_token_id

    # Example usage
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

    text = """The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout the world. This period marked a significant turning point in history; almost every aspect of daily life was influenced in some way.One of the primary factors that contributed to the Industrial Revolution was the agricultural revolution. Improvements in farming techniques and livestock breeding led to increased food production. This allowed for a surplus of food, which in turn meant that fewer people were needed to work in agriculture. As a result, many people moved to the cities to find work in the new factories.The development of new machinery and technology was another critical factor in the Industrial Revolution. Innovations such as the spinning jenny, the power loom, and the steam engine revolutionized the textile industry. These inventions allowed for the mass production of goods, which drastically reduced the cost and time required to produce textiles. The steam engine, in particular, played a crucial role in the Industrial Revolution. It was used to power machinery, locomotives, and ships, making transportation and manufacturing more efficient.
The rise of factories also had a profound impact on society. People who had once worked on farms or in small workshops now found themselves working long hours in large, noisy factories. The working conditions were often harsh, with little regard for the health and safety of the workers. Child labor was also common during this time, with young children working long hours in dangerous conditions for very low pay.The Industrial Revolution also led to significant social and economic changes. The rise of factories and the mass production of goods created a new class of wealthy industrialists and businessmen. At the same time, a new working class emerged, made up of people who worked in the factories and lived in crowded, unsanitary conditions in the cities. The gap between the rich and the poor widened, leading to social tensions and unrest.
Despite the many challenges and hardships, the Industrial Revolution also brought about many positive changes. It led to the development of new technologies and innovations that improved the quality of life for many people. It also paved the way for the modern industrial economy, with its emphasis on mass production and technological innovation.In conclusion, the Industrial Revolution was a period of significant change and transformation. It had a profound impact on almost every aspect of society, from agriculture and industry to transportation and social structures. While it brought about many challenges and hardships, it also laid the foundation for the modern world and its emphasis on technology and innovation."""
    summary = summarize(text)
    print(summary)
