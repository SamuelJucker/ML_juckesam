import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load the dataset
df = pd.read_csv('path_to_your_dolly_15k.csv')

# Preprocess the data
df = df.dropna()  # Drop rows with missing values
train_df = df.sample(frac=0.8, random_state=42)  # 80% for training
val_df = df.drop(train_df.index)  # 20% for validation

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Tokenize the inputs and labels
def tokenize_function(examples):
    return tokenizer(examples['instruction'], truncation=True, padding='max_length')

train_encodings = train_df.apply(tokenize_function, axis=1)
val_encodings = val_df.apply(tokenize_function, axis=1)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=val_encodings,
)

# Train the model
trainer.train()
