import re
import torch
from torch import nn
from transformers import Trainer
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from typing import Optional

import torch
import torch.nn as nn
import wandb

def find_prompt_indices(tokenized_sentence, prompt_tokens=(11925,5239)): # 11925 is "len" and 5239 is "text"
    prompt_indices = []
    for i, token in enumerate(tokenized_sentence):
        if token == prompt_tokens[0] and i+6 < len(tokenized_sentence) and tokenized_sentence[i+4] == prompt_tokens[1]:
            prompt_indices.extend([i+x for x in range(-1,6)])
    return prompt_indices


class LengthTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        This computes normal CE loss, without considering the prompt
        """
        # forward pass
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        # get my indexes for the prompts
        ignore_indexes = find_prompt_indices(labels.view(-1))
        indices_to_keep = [i for i in range(labels.view(-1).size(0)) if i not in ignore_indexes]
        indices_to_keep = torch.tensor(indices_to_keep, dtype=torch.long).cuda()
        new_labels = torch.index_select(labels.view(-1), 0, indices_to_keep)
        # similarly drop indexes for my logits
        new_logits = torch.index_select(logits.view(-1, logits.size(-1)), 0, indices_to_keep)
        # compute CE Loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(new_logits, new_labels)

        # log loss values
        true_labels = []
        pred_labels = []
        decoded_outputs = self.tokenizer.decode(torch.max(logits, dim=-1)[1].view(-1))
        split_sentences = decoded_outputs.split("<len>")
        for split_sentence in split_sentences:
            if "<text>" in str(split_sentence):
                first = split_sentence.split("<text>")[0].strip()
                if first.isnumeric():
                    true_labels.append(int(first))
                    generated_num_words = len(word_tokenize(split_sentence.split("<text>")[1].strip()))
                    pred_labels.append(generated_num_words)
        loss_l1loss = nn.L1Loss()
        len_loss = loss_l1loss(torch.FloatTensor(true_labels), torch.FloatTensor(pred_labels)).cuda()
        wandb.log({'loss': loss, 'len_metric': len_loss})

        return (loss, outputs) if return_outputs else loss








# self.tokenizer.decode(cool.view(-1))
#         # map to labels
#         dec_labels = self.tokenizer.decode(labels.view(-1))
#         dec_output_labels = self.tokenizer.decode(torch.max(logits, dim=-1)[1].view(-1))
#         # remove all starting prompts
#         dec_labels = ''.join([i for i in dec_labels if not i.isdigit()])
#         dec_labels = dec_labels.replace("<len>", "" )
#         dec_labels = dec_labels.replace("<text>", "" )
#         dec_output_labels = ''.join([i for i in dec_output_labels if not i.isdigit()])
#         dec_output_labels = dec_output_labels.replace("<len>", "" )
#         dec_output_labels = dec_output_labels.replace("<text>", "" )
#         # revert back to token _ids
#         dec_labels = self.tokenizer.encode(dec_labels)
#         dec_output_labels = self.tokenizer.encode(dec_output_labels)
#         # calculate loss
#         loss_fct = nn.CrossEntropyLoss()


#         return (loss, outputs) if return_outputs else loss


    # def evaluate(self, eval_dataset):
    #     breakpoint()
    #     labels = inputs.get("labels")
    #     # forward pass
    #     outputs = model(**inputs)
    #     breakpoint()
    #     logits = outputs.get("logits")
    #     # # compute CE Loss
    #     loss_fct = nn.CrossEntropyLoss()
    #     loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # # output_labels = torch.max(logits, dim=-1)[1].view(-1)
        # # flatten
        # output_labels = output_labels.view(-1)
        # # decode
        # decoded_outputs = self.tokenizer.decode(torch.max(new_logits, dim=-1)[1].view(-1))
        #self.tokenizer.decode(torch.max(logits, dim=-1)[1].view(-1))
        # # compute my lengths
        # true_labels = []
        # pred_labels = []
        # split_sentences = decoded_outputs.split("<len>")
        
        # for split_sentence in split_sentences:
        #     if "<text>" in str(split_sentence):
        #         first = split_sentence.split("<text>")[0].strip()
        #         if first.isnumeric():
        #             true_labels.append(int(first))
        #             generated_num_words = len(word_tokenize(split_sentence.split("<text>")[1].strip()))
        #             pred_labels.append(generated_num_words)
        # loss_fn = nn.L1Loss()
        # loss = loss_fn(torch.FloatTensor(true_labels), torch.FloatTensor(pred_labels)).cuda()
        # loss = torch.tensor(loss, requires_grad=True)
        



# # # logits = outputs.get("logits")
# # # output_labels = torch.max(logits, dim=-1)[1]
# # # # flatten
# # # output_labels = output_labels.view(-1)
# # # # decode
# # # decoded_labels = self.tokenizer.decode(output_labels)