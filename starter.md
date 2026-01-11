# Gut Microbiome Project - Starter Guide

> **Quick Overview**: This project uses machine learning to predict food allergy development from gut microbiome data. It leverages foundation models (ProkBERT + MicrobiomeTransformer) to create embeddings from bacterial DNA sequences, which are then classified using standard ML models.

---

## 📋 Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Scientific Background](#2-scientific-background)
3. [Architecture Overview](#3-architecture-overview)
4. [Key Components](#4-key-components)
5. [Data Pipeline](#5-data-pipeline)
6. [Getting Started](#6-getting-started)
7. [Repository Structure](#7-repository-structure)
8. [Development Workflow](#8-development-workflow)
9. [Common Tasks](#9-common-tasks)
10. [Resources & Links](#10-resources--links)

---

## 1. Project Purpose

The primary goal is to build a **binary classifier** that distinguishes between "healthy" and "allergic" subjects based on their gut microbiota composition. The project aims to:

- **Predict food allergies** before clinical symptoms appear
- **Leverage modern foundation models** for microbiome data representation
- **Enable cross-dataset generalization** through standardized embedding approaches
- **Provide reproducible ML pipelines** for microbiome research

### Target Outcomes
- Early detection of food allergy risk in infants/children
- Potential for personalized intervention strategies
- Extensible framework for other microbiome-related predictions

---

## 2. Scientific Background

### The Microbiome-Allergy Link

Research shows that gut microbiota plays a critical role in immune tolerance to dietary antigens:

| Finding | Implication |
|---------|-------------|
| **Reduced Diversity** | Allergic children have less diverse gut bacteria |
| **Depleted Protective Taxa** | Loss of beneficial bacteria like *Bifidobacterium*, *Faecalibacterium* |
| **Enriched Pro-inflammatory Taxa** | Increase in *Escherichia-Shigella*, *Ruminococcus gnavus* |
| **Temporal Predictability** | Microbial shifts are detectable months before allergy symptoms |

### Why Foundation Models?

Traditional microbiome analysis uses OTU (Operational Taxonomic Unit) abundance tables. This project goes further by:

1. **Embedding DNA sequences** using ProkBERT (a transformer trained on bacterial DNA)
2. **Aggregating OTU embeddings** using MicrobiomeTransformer (learns sample-level representations)
3. **Classification** using standard sklearn models on the learned embeddings

This approach captures more nuanced biological signals than simple abundance counts.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐             │
│  │ Sample CSV   │    │ Parquet      │    │ DNA Sequences      │             │
│  │ (sid, label) │───▶│ Mappings     │───▶│ per sample         │             │
│  └──────────────┘    │ SRS→OTU→DNA  │    │ (CSV files)        │             │
│                      └──────────────┘    └─────────┬──────────┘             │
│                                                    │                         │
│                                                    ▼                         │
│                                          ┌────────────────────┐             │
│                                          │ ProkBERT Model     │             │
│                                          │ (DNA → 384-dim     │             │
│                                          │  embeddings)       │             │
│                                          └─────────┬──────────┘             │
│                                                    │                         │
│                                                    ▼                         │
│                                          ┌────────────────────┐             │
│                                          │ DNA Embeddings H5  │             │
│                                          │ (per OTU: 384-dim) │             │
│                                          └─────────┬──────────┘             │
│                                                    │                         │
│                                                    ▼                         │
│                                          ┌────────────────────┐             │
│                                          │ MicrobiomeTransf.  │             │
│                                          │ (aggregates OTUs   │             │
│                                          │  → 100-dim sample) │             │
│                                          └─────────┬──────────┘             │
│                                                    │                         │
│                                                    ▼                         │
│                                          ┌────────────────────┐             │
│                                          │ Microbiome Emb H5  │             │
│                                          │ (per sample:       │             │
│                                          │  100-dim vector)   │             │
│                                          └─────────┬──────────┘             │
│                                                    │                         │
└────────────────────────────────────────────────────┼─────────────────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLASSIFICATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐             │
│  │ DataFrame    │    │ SKClassifier │    │ Evaluation         │             │
│  │ (embeddings  │───▶│ - LogReg     │───▶│ - ROC-AUC          │             │
│  │  + labels)   │    │ - RF         │    │ - Confusion Matrix │             │
│  │              │    │ - SVM        │    │ - Classification   │             │
│  │              │    │ - MLP        │    │   Report           │             │
│  └──────────────┘    └──────────────┘    └────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Components

### 4.1 Models

| Model | File | Purpose |
|-------|------|---------|
| **ProkBERT** | External (HuggingFace) | Encodes bacterial DNA sequences into 384-dim vectors |
| **MicrobiomeTransformer** | `modules/model.py` | Aggregates OTU embeddings into sample-level representations |
| **SKClassifier** | `modules/classifier.py` | Sklearn-based classifiers with CV and grid search |

### 4.2 Configuration

All parameters are centralized in `config.yaml`:

```yaml
data:
  dataset_path: "data_preprocessing/datasets/diabimmune/Month_9.csv"
  device: "cpu"  # cpu, cuda, or mps
  
model:
  classifier: "logreg"  # logreg, rf, svm, mlp
  use_scaler: true
  param_grids:
    logreg:
      C: [0.001, 0.01, 0.1, 1, 10]
      penalty: ["l1", "l2"]
      
evaluation:
  cv_folds: 5
  grid_search_scoring: "roc_auc"
```

### 4.3 Datasets Available

| Dataset | Description | Location |
|---------|-------------|----------|
| **DIABIMMUNE** | Finnish birth cohort, monthly samples | `datasets/diabimmune/` |
| **Goldberg** | Time-series food allergy study | `datasets/goldberg/` |
| **Tanaka** | Japanese microbiome study | `datasets/tanaka/` |
| **Gadir** | Additional allergy cohort | `datasets/gadir/` |

---

## 5. Data Pipeline

### Input Data Requirements

1. **Sample CSV** - Contains sample IDs and labels
   ```csv
   Trial,sid,label
   T1,ERS4516182,1
   T1,ERS4516184,0
   ```

2. **Parquet Mapping Files** (from Google Drive/Figshare)
   - `samples-otus-97.parquet` - Maps sample IDs → OTU IDs
   - `otus_97_to_dna.parquet` - Maps OTU IDs → DNA sequences

3. **Model Checkpoint**
   - `checkpoint_epoch_0_final_epoch3_conf00.pt` - Pre-trained MicrobiomeTransformer

### Pipeline Stages

```python
# Simplified view of data_loading.py
def load_dataset_df(config):
    # 1. Extract DNA sequences from parquet files
    extract_csv_sequences(sequences_dir, config)
    
    # 2. Generate ProkBERT embeddings (384-dim per OTU)
    generate_dna_embeddings_h5(sequences_dir, dna_embeddings_dir, ...)
    
    # 3. Generate sample embeddings (100-dim per sample)
    generate_microbiome_embeddings_h5(dna_h5, microbiome_dir, checkpoint, ...)
    
    # 4. Return DataFrame with (sid, label, embedding)
    return create_dataset_df(dataset_path, microbiome_embeddings_dir)
```

### Caching Behavior

- All intermediate artifacts are cached in H5 files
- Re-running skips already-generated embeddings
- Delete H5 files to regenerate embeddings

---

## 6. Getting Started

### Prerequisites

- Python ≥ 3.12
- `uv` package manager (recommended) or `pip`
- ~10GB disk space for data files

### Installation

```bash
# Clone the repository
git clone https://github.com/AI-For-Food-Allergies/gut_microbiome_project.git
cd gut_microbiome_project

# Create virtual environment with uv
uv sync
source .venv/bin/activate
```

### Download Required Files

1. **Datasets** (CSV files): [Google Drive](https://drive.google.com/drive/folders/1-MM3xOOhaEgILnD-D9IiLBrSBQOlz6QP)
2. **Model Checkpoint**: [Figshare](https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055)
3. **Parquet Files**: [Google Drive](https://drive.google.com/drive/folders/1d33c5JtZREoDWRAu14o-fDXOpuriuyQC)

### Quick Test

```bash
# Edit config.yaml to point to your dataset
python main.py
```

---

## 7. Repository Structure

```
gut_microbiome_project/
├── config.yaml                 # Central configuration
├── main.py                     # Entry point for training/evaluation
├── data_loading.py             # Data pipeline (embedding generation)
├── generate_embeddings.py      # Standalone batch embedding script
│
├── modules/
│   ├── model.py               # MicrobiomeTransformer class
│   └── classifier.py          # SKClassifier (LogReg, RF, SVM, MLP)
│
├── utils/
│   ├── data_utils.py          # Config loading, data prep
│   └── evaluation_utils.py    # Metrics, plotting, ResultsManager
│
├── data_preprocessing/
│   ├── datasets/              # Sample CSVs (sid, label)
│   │   ├── diabimmune/
│   │   ├── goldberg/
│   │   └── tanaka/
│   ├── mapref_data/           # Parquet mapping files
│   ├── dna_sequences/         # Generated DNA CSVs
│   ├── dna_embeddings/        # ProkBERT embeddings (H5)
│   └── microbiome_embeddings/ # MicrobiomeTransformer embeddings (H5)
│
├── example_scripts/           # Usage examples
│   ├── predict_milk.py        # Milk-feeding type prediction
│   └── predict_hla.py         # HLA risk prediction
│
├── eval_results/              # Output directory for results
│   └── {dataset}/{timestamp}/ 
│       ├── classification_report.csv
│       ├── confusion_matrix.png
│       └── roc_curve.png
│
├── pyproject.toml             # Dependencies (Python 3.12+)
├── Contributing.md            # Contribution guidelines
└── README_EMBEDDINGS.md       # Detailed embedding generation guide
```

---

## 8. Development Workflow

### Running Experiments

```python
# main.py offers multiple modes:

# Option 1: Simple evaluation (no hyperparameter tuning)
run_evaluation(config)

# Option 2: Compare multiple classifiers
run_evaluation(config, classifiers=["logreg", "rf", "svm", "mlp"])

# Option 3: Grid search with unbiased evaluation (recommended)
run_grid_search_experiment(config, classifiers=["logreg", "rf", "svm", "mlp"])
```

### Adding New Datasets

1. Prepare CSV with columns: `sid` (sample ID), `label` (0 or 1)
2. Place in `data_preprocessing/datasets/<dataset_name>/`
3. Update `config.yaml` with the new path
4. Run pipeline - embeddings generated automatically

### Adding New Classifiers

1. Add to `modules/classifier.py`:
   ```python
   elif self.classifier_type == "xgb":
       from xgboost import XGBClassifier
       return XGBClassifier(**params)
   ```
2. Add hyperparameter grid to `config.yaml`
3. Update `CLASSIFIER_NAMES` mapping

---

## 9. Common Tasks

### Generate Embeddings for a New Dataset

```bash
# Edit generate_embeddings.py with your dataset path
python generate_embeddings.py
```

### Evaluate a Single Classifier

```python
from data_loading import load_dataset_df
from modules.classifier import SKClassifier
from utils.data_utils import load_config, prepare_data

config = load_config()
dataset_df = load_dataset_df(config)
X, y = prepare_data(dataset_df)

clf = SKClassifier("logreg", config)
metrics = clf.evaluate_model(X, y, cv=5)
print(metrics.classification_report_dict)
```

### Run Hyperparameter Search

```python
clf = SKClassifier("logreg", config)
metrics = clf.grid_search_with_final_eval(
    X, y,
    param_grid={"C": [0.1, 1, 10], "penalty": ["l2"]},
    grid_search_cv=5,
    final_eval_cv=5
)
```

---

## 10. Resources & Links

### Project Links
- **GitHub**: [AI-For-Food-Allergies/gut_microbiome_project](https://github.com/AI-For-Food-Allergies/gut_microbiome_project)
- **Discord**: [huggingscience](https://discord.com/invite/VYkdEVjJ5J)
- **Data (Figshare)**: [Model and Data](https://figshare.com/articles/dataset/Model_and_Data_for_diabimmune_example/30429055)

### Scientific References
- [Micro-Modelling the Microbiome](https://the-puzzler.github.io/post.html?p=posts%2Fmicro-modelling%2Fmicro-modelling.html) - Foundation model approach
- ProkBERT: `neuralbioinfo/prokbert-mini-long` on HuggingFace

### Key Dependencies
| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework |
| `transformers` | ProkBERT model loading |
| `scikit-learn` | Classifiers and evaluation |
| `h5py` | Embedding storage |
| `pandas/numpy` | Data manipulation |

---

## 🚀 Next Steps

1. **Set up environment**: `uv sync && source .venv/bin/activate`
2. **Download data files**: Follow links in [Getting Started](#6-getting-started)
3. **Run a test**: `python main.py`
4. **Explore examples**: Check `example_scripts/` for specific use cases
5. **Read contribution guide**: See `Contributing.md` for development guidelines

---

*Last updated: January 2026*
