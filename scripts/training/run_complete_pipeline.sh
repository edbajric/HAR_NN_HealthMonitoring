#!/bin/bash
# Complete HAR Training Pipeline Runner
# This script executes preprocessing and training end-to-end

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "HAR Neural Network Training Pipeline"
echo "=========================================================="
echo ""

# Check if data files exist
if [ ! -f "train.csv" ]; then
    echo "❌ Error: train.csv not found in $SCRIPT_DIR"
    exit 1
fi

if [ ! -f "test.csv" ]; then
    echo "❌ Error: test.csv not found in $SCRIPT_DIR"
    exit 1
fi

echo "✓ Data files found"
echo ""

# Step 1: Run preprocessing (Python)
echo "Step 1: Running preprocessing..."
python3 <<'PREPROCESSING_SCRIPT'
import sys
sys.path.insert(0, '/Users/edinabajric/Desktop/Codes/AI_ML/HAR_NN_HealthMonitoring/scripts/training')

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json

print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Extract features
feature_columns = [col for col in train_df.columns if col not in ['Activity', 'subject']]
X_train = train_df[feature_columns].values
y_train = train_df['Activity'].values
X_test = test_df[feature_columns].values
y_test = test_df['Activity'].values

print(f"Data loaded: {X_train.shape[0]} training, {X_test.shape[0]} test")

# Scale using MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data and scaler params
train_df_processed = train_df.copy()
train_df_processed[feature_columns] = X_train_scaled
train_df_processed.to_csv('train_processed.csv', index=False)

test_df_processed = test_df.copy()
test_df_processed[feature_columns] = X_test_scaled
test_df_processed.to_csv('test_processed.csv', index=False)

scaler_params = {
    'min': scaler.data_min_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': feature_columns
}

with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

print("✓ Preprocessing complete")
print(f"  - Saved train_processed.csv")
print(f"  - Saved test_processed.csv")
print(f"  - Saved scaler_params.json")
PREPROCESSING_SCRIPT

echo ""

# Step 2: Run training
echo "Step 2: Running training..."
python3 har_training.py \
    --train_path train.csv \
    --test_path test.csv \
    --output ../../public/model/weights.json \
    --output_dir ../../public/model

echo ""
echo "=========================================================="
echo "✓ Pipeline Complete!"
echo "=========================================================="
echo ""
echo "Output files:"
echo "  ✓ ../../public/model/weights.json"
echo "  ✓ ../../public/model/confusion_matrix.json"
echo "  ✓ ../../public/model/training_history.json"
echo ""
echo "Next: Run 'npm run dev' to start the web app"
