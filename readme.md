# ML-Based Prediction of Food Allergy Development From Gut Microbiome Data

## Project Overview

This project aims to develop a robust machine learning classifier that can predict the development of food allergies by analyzing an individual's gut microbiome data. By leveraging the latest advancements in microbiome foundation models, we seek to transform the landscape of food allergy prediction and prevention.

The core goal is to build a **binary classifier** that can distinguish between "healthy" and "allergic" subjects based on their gut microbiota composition. Future iterations will explore extending this model to predict the risk of developing specific types of food allergies.

## Motivation: The Microbiome-Allergy Link

Accumulating scientific evidence highlights the critical role of the gut microbiota in shaping immune tolerance to dietary antigens. Key findings that motivate this research include:

*   **Reduced Diversity:** Allergic children consistently exhibit reduced microbial diversity compared to healthy controls.
*   **Taxa Shifts:** There is a notable depletion of protective taxa (e.g., *Bifidobacterium*, *Faecalibacterium*, butyrate-producing *Clostridia*) and an enrichment of pro-inflammatory bacteria (e.g., *Escherichia-Shigella*, *Ruminococcus gnavus*).
*   **Predictive Potential:** These microbial shifts are often detectable months before the clinical manifestation of food allergies, suggesting a strong predictive potential for early intervention.

## Methodology

Our approach is built on a modern, two-stage machine learning architecture designed to handle the complexity and high dimensionality of microbiome data:

1.  **Microbiome Embedding Model:** We utilize a recently developed **foundation model** for microbiome data. This model serves as a backbone, extracting rich, low-dimensional, and meaningful representations (embeddings) from raw 16S rRNA or shotgun metagenomics data.
2.  **Classifier Head:** The extracted embeddings are then fed into a simpler machine learning model, such as **Logistic Regression** or a similar classifier, to perform the final binary prediction.

### Evaluation Strategy

To ensure the model is robust and generalizable, we employ a rigorous evaluation strategy:

*   **Standard Metrics:** Performance is assessed using standard classification metrics (e.g., AUC, F1-score, Accuracy) with cross-validation.
*   **Cross-Dataset Validation:** We implement a **leave-one-dataset-out** validation strategy to test the model's ability to generalize across different cohorts, sequencing protocols, and study conditions, which is crucial for real-world applicability.

## Repository Structure

The project follows a modular structure to separate concerns and facilitate collaboration:

| Directory/File | Purpose |
| :--- | :--- |
| `data_preprocessing/` | Scripts for cleaning, transforming, and preparing raw microbiome data. |
| `modules/` | Reusable classes and functions for the model architecture (e.g., the `MicrobiomeTransformer` wrapper). |
| `evaluation/` | Scripts for model performance assessment, metric calculation, and visualization. |
| `train.py` | The main entry point for running the model training pipeline. |
| `main.py` | The overall execution script for the entire project workflow. |
| `data_loading.py` | Unified data loading pipeline with automatic artifact generation (DNA CSVs, embeddings H5) and PyTorch DataLoader integration. |
| `config.yaml` | Centralized YAML configuration file for all run parameters (data paths, model settings, evaluation metrics). |
| `utils.py` | Helper functions including configuration loading from YAML. |

## Getting Started

To set up the project environment and begin contributing:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/AI-For-Food-Allergies/gut_microbiome_project.git
    cd gut_microbiome_project
    ```
2.  **Install dependencies:**
    ```bash
    # Using pip
    pip install -e .
    
    # Or using uv (if installed)
    uv sync
    ```
3.  **Download the Microbiome Transformer checkpoint:**
    
    The pre-trained Microbiome Transformer model checkpoint can be downloaded from Figshare:
    
    [Model and Data for diabimmune example](https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055?file=58993825)

4.  **Configure your project:**
    
    All run parameters are managed through `config.yaml`. This includes:
    - Data paths (dataset location, checkpoint paths)
    - Model settings (classifier type, hyperparameters)
    - Evaluation parameters (metrics, cross-validation settings)
    
    Edit `config.yaml` to customize these parameters for your needs:
    ```yaml
    data:
      dataset_path: "path/to/your/data"
    
    model:
      classifier: "linear regression"
      classifier_params:
        max_iter: 1000
        solver: "lbfgs"
    
    evaluation:
      metrics: ["accuracy", "precision", "recall", "f1", "roc_auc"]
      cv_folds: 5
    ```

## Data Loading Pipeline

The `data_loading.py` module provides a unified pipeline for loading microbiome data ready for training. It automatically handles artifact generation (DNA sequences and embeddings) and provides PyTorch DataLoaders with proper batching and masking for variable-length sequences.

### Quick Start

```python
from pathlib import Path
from data_loading import get_dataloader

# Simple usage - everything handled automatically
dataloader = get_dataloader(
    Path("data_preprocessing/datasets/sample_data.csv"),
    batch_size=4,
    shuffle=True
)

# Iterate over batches
for batch in dataloader:
    embeddings = batch['embeddings']  # (batch_size, max_otus, 384)
    labels = batch['labels']          # (batch_size,)
    mask = batch['mask']              # (batch_size, max_otus) - True for valid, False for padding
    sids = batch['sids']               # List[str] - sample IDs
```

### Features

- **Automatic Artifact Generation**: Automatically generates DNA CSV files and embeddings H5 when missing
- **Caching**: Only generates missing artifacts, reuses existing ones
- **Variable-Length Sequences**: Handles samples with different numbers of OTUs via padding and masking
- **Config-Driven**: All paths and settings configurable via `config.yaml`

### Data Format

Your sample CSV file should have two columns:
- **Sample ID column** (e.g., `sid`, `SID`, `srs_id`): Unique identifier for each sample
- **Label column** (e.g., `label`, `Label`, `y`): Binary label (0/1) for classification

Example:
```csv
sid,label
DRS061545,0
DRS061546,1
DRS061547,0
```

### Configuration

Configure data paths and embedding settings in `config.yaml`:

```yaml
data:
  # Parquet files mapping SRS → OTU → DNA
  srs_to_otu_parquet: "data_preprocessing/mapref_data/samples-otus-97.parquet"
  otu_to_dna_parquet: "data_preprocessing/mapref_data/otus_97_to_dna.parquet"
  
  # Output directories
  dna_csv_dir: "data_preprocessing/dna_sequences"
  embeddings_h5: "data_preprocessing/dna_embeddings/prokbert_embeddings.h5"
  
  # Embedding generation settings
  embedding_model: "neuralbioinfo/prokbert-mini-long"
  batch_size_embedding: 32
  device: "cpu"  # cpu, cuda, or mps
```

### Pipeline Workflow

1. **Input**: Sample CSV file with SID and label columns
2. **Step 1**: Check if DNA CSV files exist for each sample, generate missing ones
3. **Step 2**: Check if embeddings H5 exists, generate from DNA CSVs if missing
4. **Step 3**: Load embeddings and labels, create PyTorch Dataset
5. **Output**: DataLoader with batched data ready for training

### Advanced Usage

```python
from data_loading import MicrobiomeDataset, load_dataset

# Load dataset without DataLoader
X_list, y_list, sids_list = load_dataset(
    Path("data_preprocessing/datasets/sample_data.csv")
)

# Create custom Dataset
dataset = MicrobiomeDataset(
    Path("data_preprocessing/datasets/sample_data.csv"),
    embeddings_h5_path=None  # Auto-generates if None
)
```

For more details, see the docstrings in `data_loading.py`.

    

## 🤝 Contributing

We welcome contributions from researchers, data scientists, and developers!

Please refer to our dedicated **[Contributing Guide](https://github.com/AI-For-Food-Allergies/gut_microbiome_project/blob/master/Contributing.md)** for detailed instructions on:

*   Setting up your development environment.
*   The current development roadmap and open issues.
*   Guidelines for submitting pull requests and writing clean code.

**Want to be assigned to an issue?** Join the [huggingscience Discord server](https://discord.com/invite/VYkdEVjJ5J) and communicate your interest in the relevant channel!

Thank you for helping us advance the prediction of food allergies!
