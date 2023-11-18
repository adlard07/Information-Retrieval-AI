import sys
import json
import datasets
from dataclasses import dataclass

from src.exception import CustomException
from src.utils import remake_data
from src.logger import logging


# id, title, context, question, answer
@dataclass
class DataIngestion:
    def __init__(self):
        self.train_data_path = 'data/train-v1.1.json'
        self.dev_data_path = 'data/dev-v1.1.json'
    
    def get_data(self):
        try:
            train = open(self.train_data_path)
            dev = open(self.dev_data_path)
            training_data = json.load(train)
            deving_data = json.load(dev)
            train.close()
            dev.close()
            
            train_dataset = remake_data(training_data)
            dev_dataset = remake_data(deving_data)
            return(
                train_dataset,
                dev_dataset
            )
        except Exception as e:
            raise CustomException(e, sys)
        
class DataFrameMaker:
    def __init__(self):
        self.train_data = None
        self.dev_data = None

    def get_df(self, train_dataset, dev_dataset):
        try:
            train_df = datasets.Dataset.from_dict(train_dataset)
            dev_df = datasets.Dataset.from_dict(dev_dataset)
            
            data = datasets.DatasetDict({
                'train' : train_df,
                'val' : dev_df
            })
            
            return data
        except Exception as e:
            raise CustomException(e, sys)

        
        
if __name__=='__main__':
    ingestion = DataIngestion()
    train_dataset, dev_dataset = ingestion.get_data()
    
    df_maker = DataFrameMaker()
    data = df_maker.get_df(train_dataset, dev_dataset)

    logging.info("Data Ingestion Complete!")
    print('Data Ingestion Complete!')
    
    print(data['train'][0])
    print(data)
    