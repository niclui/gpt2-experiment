import re
import torch
from torch import nn
from transformers import Trainer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

class LengthTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        output_labels = torch.max(logits, dim=-1)[1]
        # flatten
        output_labels = output_labels.view(-1)
        # decode
        decoded_outputs = self.tokenizer.decode(output_labels)
        # compute my lengths
        true_labels = []
        pred_labels = []
        split_sentences = decoded_outputs.split("<len>")
        
        for split_sentence in split_sentences:
            if "<text>" in str(split_sentence):
                first = split_sentence.split("<text>")[0].strip()
                if first.isnumeric():
                    true_labels.append(int(first))
                    generated_num_words = len(word_tokenize(split_sentence.split("<text>")[1].strip()))
                    pred_labels.append(generated_num_words)
        loss_fn = nn.L1Loss()
        loss = loss_fn(torch.FloatTensor(true_labels), torch.FloatTensor(pred_labels)).cuda()
        loss = torch.tensor(loss, requires_grad=True)
        return (loss, outputs) if return_outputs else loss



# logits = outputs.get("logits")
# output_labels = torch.max(logits, dim=-1)[1]
# # flatten
# output_labels = output_labels.view(-1)
# # decode
# decoded_labels = self.tokenizer.decode(output_labels)