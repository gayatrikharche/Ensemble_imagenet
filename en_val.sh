#!/bin/bash
echo "Hello OSPool from Job $1 running on `hostname`"

pip install scipy

# Create data directory and move the training data tar file
mkdir -p data
mv subset_model1.pt data/
mv subset_model2.pt data/
mv subset_model3.pt data/
mv subset_model4.pt data/
mv subset_model5.pt data/

# Run the PyTorch model
python ensemble_val.py

# Clean up
rm -rf data
