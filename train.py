# Training scripts 
from modules.classifier import MicrobiomeClassifier

def train_classifier(classifier: MicrobiomeClassifier, data):
    """
    Train a classifier on the data
    Args:
        classifier: MicrobiomeClassifier
        data: DataFrame with id, label, and microbiome embedding
    Returns:
        Trained classifier
    """
    ...