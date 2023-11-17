import os
import pandas as pd
import sys

from src.exception import CustomException
from src.logger import logging


# reads pdfs
def remake_data(data):
    try:
        titles = []
        contexts = []
        answers = []
        ids = []
        questions = []
        for i in range(len(data)):
            for each in range(len(data['data'][0]['paragraphs'])):
                title = data['data'][0]['title']
                titles.append(title)
                
                context = data['data'][0]['paragraphs'][each]['context']
                contexts.append(context)
        
                answer = data['data'][0]['paragraphs'][each]['qas'][0]['answers']
                answers.append(answer)
                
                id = data['data'][0]['paragraphs'][each]['qas'][0]['id']
                ids.append(id)
                
                question = data['data'][0]['paragraphs'][each]['qas'][0]['question']
                questions.append(question)
        
        datagram = {}
        datagram['id'] = ids
        datagram['title'] = titles
        datagram['context'] = contexts
        datagram['question'] = questions
        datagram['answers'] = answers
            
        return datagram
    except Exception as e:
        raise CustomException(e, sys)

        
# saves model and tokenizer
def save_model(model, model_path):
    try:
        if os.path.exists(model_path):
            pass
        else:    
            model.save_pretrained(model_path)
            logging.info('Model Saved!')
            print('Model Saved!')
            
    except Exception as e:
        raise CustomException(e, sys)
    
def find_start_end(example, tokenized_example, sequence_ids):
    try:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        answers = example['answers']
        start_char = answers["answer_start"]
        end_char = start_char + len(answers["text"][0])

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(tokenized_example["input_ids"][0]) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        offsets = tokenized_example["offset_mapping"][0]
        if (
            offsets[token_start_index][0] <= start_char
            and offsets[token_end_index][1] >= end_char
        ):
            while (
                token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            start_position = token_start_index - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_position = token_end_index + 1
            
            return (
                answers,
                start_position,
                end_position
                )
    except Exception as e:
        raise CustomException(e, sys)


def prepare_train_features(examples, tokenizer, pad_on_right,  max_length, doc_stride):
    try:
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized_examples.sequence_ids(i)
            
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers[0]['answer_start']
                end_char = start_char + len(answers[0]["text"])

                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
    except Exception as e:
        raise CustomException(e, sys)
