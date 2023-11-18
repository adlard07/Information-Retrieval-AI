import os
import sys
import torch
from torch.nn.functional import pad
import tensorflow as tf
from datasets import Dataset
from dataclasses import dataclass
import transformers
from transformers import TrainingArguments, Trainer, create_optimizer

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
        get_model = GetModels()
        self.model, _ = get_model.get_model_object()
        self.tokenizer, _ = get_model.get_data_tokenizer_object()
            
        # Tokenized data
        data = DataTokenizer()
        self.tokenized_datasets = data.initiate_data_tokenization()
            
        # Model Parameters
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.epochs = 2
        
        #huggingface model upload
        self.push_to_hub_model_id = f"chatGPT-clone-finetuned-squad"
        
    def model_params(self):
        train_set = Dataset.to_tf_dataset(
            self.tokenized_datasets["train"],
            shuffle=True,
            batch_size=self.batch_size,
        )

        validation_set = Dataset.to_tf_dataset(
            self.tokenized_datasets["val"],
            shuffle=False,
            batch_size=self.batch_size,
        )

        total_train_steps = len(train_set) * self.epochs
        
        optimizer, schedule = create_optimizer(
            init_lr=self.learning_rate, num_warmup_steps=0, num_train_steps=total_train_steps
        )

        # Compile model
        # self.model.compile(optimizer=optimizer, jit_compile=True, metrics=["accuracy"])
        
        return (
            self.model,
            train_set,
            validation_set, self.epochs
            )
        
if __name__=='__main__':

    model_trainer = ModelTrainer()
    model, train_set, val_set, epochs = model_trainer.model_params()
    print(train_set)
    
    model.fit(train_set, validation_data=val_set, epochs=epochs)
    
    # padded_sequences, trainer = model_trainer.model_initialize(train_encodings)
    # trainer.train() 
    # print("Model Training Complete")
    # print(padded_sequences)
    
    # save_model(model)
    # print("Model Saved!")