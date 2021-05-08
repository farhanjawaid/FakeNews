import json
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

import numpy as np


with open("config.json") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):
        self.device = "cpu"
        default_model_config = AutoConfig.from_pretrained(config["DEFAULT_MODEL_NAME"], num_labels=2)

        default_model = AutoModelForSequenceClassification.from_config(config=default_model_config)
        
        self.default_tokenizer = AutoTokenizer.from_pretrained(config["DEFAULT_MODEL_NAME"], use_fast=True)
        
        default_model.load_state_dict(
            torch.load(config["DEFAULT_MODEL"], map_location=self.device)
        )
        default_model = default_model.eval()
        self.default_model = default_model.to(self.device)

    def multilingual_news_detect(self, text):        
        prepared_input = self.default_tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')
        self.default_model = self.default_model.to('cpu')
        self.default_model.eval()
        model_output = self.default_model(**prepared_input)
        prediction = np.argmax(model_output.logits[0].detach().numpy())
        print(prediction)
        
        return (
            config["DEFAULT_CLASS_NAMES"][prediction]
        )


model = Model()


def get_model():
    return model

