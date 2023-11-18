import sys
from dataclasses import dataclass
from functools import partial

from src.components.data_ingestion import DataIngestion
from src.components.get_tokenizer_model import GetModels
from src.components.data_ingestion import DataFrameMaker

from src.logger import logging
from src.exception import CustomException
from src.utils import find_start_end, prepare_train_features


@dataclass
class DataTokenizer:
    def __init__(self):
        get_models = GetModels()
        self.tokenizer, _ = get_models.get_data_tokenizer_object()
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = self.tokenizer.eos_token

            
        self.max_length = 384  
        self.doc_stride = 128  
            
            
    def initiate_data_tokenization(self):
        try:
            ingestion = DataIngestion()
            train_dataset, dev_dataset = ingestion.get_data()

            df_maker = DataFrameMaker()
            datasets = df_maker.get_df(train_dataset, dev_dataset)
            
            for i, example in enumerate(datasets["train"]):
                if len(self.tokenizer(example["question"], example["context"])["input_ids"]) > 384:
                    break
            example = datasets["train"][i]
            
            tokenized_datasets = datasets.map(
                partial(prepare_train_features, tokenizer=self.tokenizer, pad_on_right=self.tokenizer.eos_token, max_length=self.max_length, doc_stride=self.doc_stride),
                batched=True,
                remove_columns=datasets["train"].column_names
                )
            
            return (
                tokenized_datasets
                )
                       
        except Exception as e:
            raise CustomException(e, sys)


if __name__=='__main__':
    transform = DataTokenizer()
    tokenized_datasets = transform.initiate_data_tokenization()
    print(tokenized_datasets["train"]['start_positions'])