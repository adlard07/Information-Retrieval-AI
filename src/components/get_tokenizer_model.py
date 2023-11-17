import os
import sys
import tensorflow as tf
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.logger import logging
from src.utils import save_model
from src.exception import CustomException


class GetModels:
    def __init__(self):
        self.context_path = 'data/content/context.txt'
        self.tokenizer_path = 'artifacts/tokenizer'
        self.model_path = 'artifacts/model'
        
    
    def get_data_tokenizer_object(self):
        try:
            if os.path.exists(self.tokenizer_path):
                tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
                logging.info('Tokenizer available!')
                print('Tokenizer available!') 
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                logging.info('Downloaded Tokenizer!')
                print('Downloaded Tokenizer!')
                save_model(tokenizer, self.tokenizer_path)
            return (
                tokenizer, 
                self.tokenizer_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)
                
                
    def get_model_object(self):
        try:
            if os.path.exists(self.model_path):
                model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
                logging.info('Model available!')
                print('Model available!')
            else:
                model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                logging.info('Downloaded Model!')
                print('Downloaded Model!')
                save_model(model, self.model_path)
            return (
                model, 
                self.model_path
                )
             
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=='__main__':
    get_models = GetModels()
    print('Processing...')
    tokenizer, tokenizer_path = get_models.get_data_tokenizer_object()
    model, model_path = get_models.get_model_object()
    print('Process Complete!')
    print('\n')