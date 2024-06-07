import os
import kaggle
import pandas as pd

# Define paths
input_data_path = r"C:\Users\jucke\Desktop\ML2\ML_TEST\datapipeline\input\data"
log_path = r"C:\Users\jucke\Desktop\ML2\ML_TEST\Log"

# Download dataset from Kaggle
os.makedirs(input_data_path, exist_ok=True)
kaggle.api.dataset_download_files('databricks/databricks-dolly-15k', path=input_data_path, unzip=True)

# Load the dataset
dataset_path = os.path.join(input_data_path, 'databricks-dolly-15k.csv')
df = pd.read_csv(dataset_path)

# Inspect the data
inspection_report = df.describe(include='all')

# Save inspection report
inspection_report_path = os.path.join(log_path, 'data_inspection_report.csv')
inspection_report.to_csv(inspection_report_path)

print(f"Data inspection report saved to: {inspection_report_path}")
