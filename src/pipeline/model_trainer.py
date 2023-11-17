import os
import sys
import torch
from torch.nn.functional import pad
import tensorflow as tf
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, AdamWeightDecay

from src.logger import logging
from src.components.get_tokenizer_model import GetModels
from src.utils import save_model
from src.exception import CustomException
from src.components.data_tokenization import DataTokenizer
from huggingface_hub import notebook_login
notebook_login()


@dataclass
class ModelTrainer:
    def __init__(self):
        # Model & Tokenizer
        self.model_path = 'artifacts/model'
        self.tokenizer_path = 'artifacts/tokenizer'
        get_model = GetModels()
        if os.path.exists(self.model_path):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            self.model, _ = get_model.get_model_object()
            
        # Model Parameters
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.optimizer = AdamWeightDecay(learning_rate=self.learning_rate, weight_decay_rate=self.weight_decay)
        
        
            
    def compilor(self):
        try:
            self.model.compile(optimizer=self.optimizer, jit_compile=True)
            return (
                self.model
            )
        except Exception as e:
            raise CustomException(e, sys)

    
    def model_initialize(self, training_dataset):
        try:
            training_args = TrainingArguments(
                output_dir="artifacts/output",
                evaluation_strategy = "epoch",
                learning_rate=2e-5,
                weight_decay=0.01,
                push_to_hub=True,
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
            )
            return (
                trainer
            )
                    
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__=='__main__':
    transform = DataTokenizer()
    train_encodings, _ = transform.initiate_data_tokenization()
    print("Training Data initiated...")

    model_trainer = ModelTrainer()
    model = model_trainer.compilor()
    
    
    # padded_sequences, trainer = model_trainer.model_initialize(train_encodings)
    # trainer.train() 
    # print("Model Training Complete")
    # print(padded_sequences)
    
    # save_model(model)
    # print("Model Saved!")