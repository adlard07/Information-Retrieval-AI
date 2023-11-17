import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.components.get_tokenizer_model import GetModels
from src.exception import CustomException
from src.logger import logging


        
class GenerateText:
    def __init__(self):
        self.model_path = 'artifacts/model'
        self.tokenizer_path = 'artifacts/tokenizer'
        get_model = GetModels()
        
        if os.path.exists(self.tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer, _ = get_model.get_data_tokenizer_object()
            
        if os.path.exists(self.model_path):
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        else:
            self.model, _ = get_model.get_model_object()
        
        
            
    def generate_text(self, input_text):
        try:
            output = []
            for step in range(5):
                inputs = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')

                bot_input_ids = torch.cat([chat_history_ids, inputs], dim=-1) if step > 0 else inputs
                
                chat_history_ids = self.model.generate(bot_input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
                
                output.append(self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

            
            return (
                self.model, 
                self.tokenizer, 
                f'>> Model output: {output[0]}',
                    )
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__=='__main__':
    
    generate = GenerateText()
    input_text = input('>> Enter text: ')
    model, tokenizer, generated_text = generate.generate_text(input_text)
    print(generated_text)
    print('Generation complete!')