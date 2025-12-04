# Main script to run train + evaluation

from data_loading import load_dataset_df
from modules.classifier import MicrobiomeClassifier
from train import train_classifier
from evaluation.evaluation import evaluate_classifier
from utils import load_config



if __name__ == "__main__":
    # load config
    config = load_config()
    
    # load data
    data = load_dataset_df(config)
    # load model
    model = MicrobiomeClassifier()
    # train model
    train_classifier()
    # evaluate model
    evaluate_classifier()
    # save model
    save_model()
    ...