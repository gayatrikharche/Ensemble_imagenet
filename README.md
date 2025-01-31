## Running the Experiment for Different ImageNet Subsets  

This experiment uses 5 subsets of ImageNet (`Imagenet_subset01.tar` to `Imagenet_subset05.tar`). To run the model on each subset, follow these steps:  

### 1. Modify the HTCondor Submit File  
In your submit file, update `transfer_input_files` to include the correct subset:  

```plaintext
transfer_input_files = main3.py, osdf:///ospool/ap20/data/gayatri.kharche/Imagenet_subset01.tar
transfer_output_files = subset_model2.pt
```

Replace `Imagenet_subset01.tar` with `Imagenet_subset02.tar`, `Imagenet_subset03.tar`, etc., for different runs.

### 2. Update the Wrapper Script  
Ensure the wrapper script moves and extracts the dataset before training:  

```bash
mv Imagenet_subset01.tar data/
cd data

# Extract the training data tar file
tar -xvf Imagenet_subset01.tar
rm -f Imagenet_subset01.tar
```

Replace `Imagenet_subset01.tar` with the correct subset filename in each run.

### 3. Modify `main3.py` to Save Models with Different Names  
Update line 119 in `main3.py` to ensure each subset's model is saved separately:  

```python
torch.save(model.state_dict(), "subset_model2.pt")
```

Change `"subset_model2.pt"` to match the subset number (e.g., `subset_model01.pt`, `subset_model02.pt`, etc.).

### 4. Run for All Subsets  
Repeat the process for `Imagenet_subset01.tar` to `Imagenet_subset05.tar` to save models for all subsets.

### 5. Run the Code  
To execute the experiment, run the following command in the current directory:  

```bash
condor_submit en.sub
```

---

