import torch.nn as nn
from transformers import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.extra_modules = nn.ModuleDict({
            'fcn1': nn.Linear(768, 30),  # The hidden_size of bert-base-uncased model is 768
            'relu': nn.ReLU(),
            'fcn2': nn.Linear(30, 1),

        })

    def forward(self, input_sentences, input_sentences_2):
        tokenized_input = self.tokenizer(input_sentences, input_sentences_2, add_special_tokens=True, padding=True, return_tensors='pt')
        output = self.bert(input_ids=tokenized_input["input_ids"].to('cuda'), token_type_ids=tokenized_input["token_type_ids"].to('cuda'), attention_mask=tokenized_input["attention_mask"].to('cuda'))
        (last_hidden_state, _) = output
        # last_hidden_state: (batch_size, sequence_length, hidden_size)
        cls_hidden_states = last_hidden_state[:, 0, :]
        # cls_hidden_states: (batch_size, hidden_size)

        x = self.extra_modules['fcn1'](cls_hidden_states)
        x = self.extra_modules['relu'](x)
        x = self.extra_modules['fcn2'](x)

        # x : (batch_size, 1)
        out = x.squeeze(1)

        # out: (batch_size.)
        return out









