#!/bin/bash
echo "Hello OSPool from Job $1 running on `hostname`"

pip install scipy

# Create data directory and move the training data tar file
mkdir -p data
mv subset1 data/
cd data

# Extract individual class tar files (suppress output)
find . -name "*.tar" | while read NAME ; do
    mkdir -p "${NAME%.tar}"
    tar -xvf "${NAME}" -C "${NAME%.tar}" > /dev/null 2>&1
    rm -f "${NAME}"
done
cd ..

# Run the PyTorch model
python main3.py --save-model --epochs 5

# Clean up
rm -rf data

