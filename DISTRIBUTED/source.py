!pip install matplotlib
!pip install seaborn

import transformers
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter("ignore")

from datasets import load_dataset
dataset = load_dataset("squad")
dataset

# Print examples
print(dataset["train"][0])
print(dataset["validation"][0])

dataset["train"].filter(lambda x: len(x["answers"]["text"]) != 1)

dataset["validation"].filter(lambda x: len(x["answers"]["text"]) != 1)

## Lets sample some dataset so that we can reduce training time.
dataset["train"] = dataset["train"].select([i for i in range(1000)])
dataset["validation"] = dataset["validation"].select([i for i in range(250)])
dataset

from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#trained_checkpoint = "distilbert-base-uncased"
#tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)

context = dataset["train"][0]["context"]
question = dataset["train"][0]["question"]
answer = dataset["train"][0]["answers"]["text"]


inputs = tokenizer(
    question,
    context,
    max_length=160,
    truncation="only_second",  # only to truncate context
    stride=70,  # no of overlapping tokens  between concecute context pieces
    return_overflowing_tokens=True,  #to let tokenizer know we want overflow tokens
)


print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")

print('Question: ',question)
print(' ')
print('Context : ',context)
print(' ')
print('Answer: ', answer)
print('--'*25)

for i,ids in enumerate(inputs["input_ids"]):
    print('Context piece', i+1)
    print(tokenizer.decode(ids[ids.index(102):]))
    print(' ')
    


from transformers import AutoTokenizer

del tokenizer
trained_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)

def train_data_preprocess(examples):
    
    """
    generate start and end indexes of answer in context
    """
    
    def find_context_start_end_index(sequence_ids):
        """
        returns the token index in whih context starts and ends
        """
        token_idx = 0
        while sequence_ids[token_idx] != 1:  #means its special tokens or tokens of queston
            token_idx += 1                   # loop only break when context starts in tokens
        context_start_idx = token_idx
    
        while sequence_ids[token_idx] == 1:
            token_idx += 1
        context_end_idx = token_idx - 1
        return context_start_idx,context_end_idx  
    
    
    questions = [q.strip() for q in examples["question"]]
    context = examples["context"]
    answers = examples["answers"]
    
    inputs = tokenizer(
        questions,
        context,
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,  #returns id of base context
        return_offsets_mapping=True,  # returns (start_index,end_index) of each token
        padding="max_length"
    )


    start_positions = []
    end_positions = []

    
    for i,mapping_idx_pairs in enumerate(inputs['offset_mapping']):
        context_idx = inputs['overflow_to_sample_mapping'][i]
    
        # from main context
        answer = answers[context_idx]
        answer_start_char_idx = answer['answer_start'][0]
        answer_end_char_idx = answer_start_char_idx + len(answer['text'][0])

    
        # now we have to find it in sub contexts
        tokens = inputs['input_ids'][i]
        sequence_ids = inputs.sequence_ids(i)
   
        # finding the context start and end indexes wrt sub context tokens
        context_start_idx,context_end_idx = find_context_start_end_index(sequence_ids)
    
        #if the answer is not fully inside context label it as (0,0)
        # starting and end index of charecter of full context text
        context_start_char_index = mapping_idx_pairs[context_start_idx][0]
        context_end_char_index = mapping_idx_pairs[context_end_idx][1]
    

        #If the answer is not fully inside the context, label is (0, 0)
        if (context_start_char_index > answer_start_char_idx) or (
            context_end_char_index < answer_end_char_idx):
            start_positions.append(0)
            end_positions.append(0)
    
        else:

            # else its start and end token positions
            # here idx indicates index of token
            idx = context_start_idx
            while idx <= context_end_idx and mapping_idx_pairs[idx][0] <= answer_start_char_idx:
                idx += 1
            start_positions.append(idx - 1)  
        

            idx = context_end_idx
            while idx >= context_start_idx and mapping_idx_pairs[idx][1] > answer_end_char_idx:
                idx -= 1
            end_positions.append(idx + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
    
train_sample = dataset["train"].select([i for i in range(200)])
    
train_dataset = train_sample.map(
    train_data_preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)

len(dataset["train"]),len(train_dataset)

def print_context_and_answer(idx,mini_ds=dataset["train"]):
    
    print(idx)
    print('----')
    question = mini_ds[idx]['question']
    context = mini_ds[idx]['context']
    answer = mini_ds[idx]['answers']['text']
    print('Theoretical values :')
    print(' ')
    print('Question: ')
    print(question)
    print(' ')
    print('Context: ')
    print(context)
    print(' ')
    print('Answer: ')
    print(answer)
    print(' ')
    answer_start_char_idx = mini_ds[idx]['answers']['answer_start'][0]
    answer_end_char_idx = answer_start_char_idx + len(mini_ds[idx]['answers']['text'][0])
    print('Start and end index of text: ',answer_start_char_idx,answer_end_char_idx)
    print('----'*20)
    print('Values after tokenization:')
    

    #answer
    sep_tok_index = train_dataset[idx]['input_ids'].index(102) #get index for [SEP]
    question_ = train_dataset[idx]['input_ids'][:sep_tok_index+1]
    question_decoded = tokenizer.decode(question_) 
    context_ = train_dataset[idx]['input_ids'][sep_tok_index+1:]
    context_decoded = tokenizer.decode(context_) 
    start_idx = train_dataset[idx]['start_positions']
    end_idx = train_dataset[idx]['end_positions']
    answer_toks = train_dataset[idx]['input_ids'][start_idx:end_idx]
    answer_decoded = tokenizer.decode(answer_toks)
    print(' ')
    print('Question: ')
    print(question_decoded)
    print(' ')
    print('Context: ')
    print(context_decoded)
    print(' ')
    print('Answer: ')
    print(answer_decoded)
    print(' ')
    print('Start pos and end pos of tokens: ',train_dataset[idx]['start_positions'],train_dataset[idx]['end_positions'])
    print('____'*20)
    
    
print_context_and_answer(0)
print_context_and_answer(1)
print_context_and_answer(2)
print_context_and_answer(3)

from transformers import AutoTokenizer

def preprocess_validation_examples(examples):
    """
    preprocessing validation data
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")

    base_ids = []

    for i in range(len(inputs["input_ids"])):
        
        # take the base id (ie in cases of overflow happens we get base id)
        base_context_idx = sample_map[i]
        base_ids.append(examples["id"][base_context_idx])
        
        # sequence id indicates the input. 0 for first input and 1 for second input
        # and None for special tokens by default
        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        # for Question tokens provide offset_mapping as None
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["base_id"] = base_ids
    return inputs


# del tokenizer

trained_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)

data_val_sample = dataset["validation"].select([i for i in range(100)])
eval_set = data_val_sample.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)
len(eval_set)

import torch
from transformers import DistilBertForQuestionAnswering

# del tokenizer
# take a small sample

eval_set_for_model = eval_set.remove_columns(["base_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

checkpoint =  "distilbert-base-uncased"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device used: {}".format(device)) #to know if we are on a gpu or a cpu
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}

model = DistilBertForQuestionAnswering.from_pretrained(checkpoint).to(
    device
)


with torch.no_grad():
    outputs = model(**batch)
    
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

start_logits.shape,end_logits.shape

!pip install evaluate

import numpy as np
import collections
import evaluate

def predict_answers_and_evaluate(start_logits,end_logits,eval_set,examples):
    """
    make predictions 
    Args:
    start_logits : strat_position prediction logits
    end_logits: end_position prediction logits
    eval_set: processed val data
    examples: unprocessed val data with context text
    """
    # appending all id's corresponding to the base context id
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(eval_set):
        example_to_features[feature["base_id"]].append(idx)

    n_best = 20
    max_answer_length = 30
    predicted_answers = []

    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # looping through each sub contexts corresponding to a context and finding
        # answers
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = eval_set["offset_mapping"][feature_index]
        
            # sorting the predictions of all hidden states and taking best n_best prediction
            # means taking the index of top 20 tokens
            start_indexes = np.argsort(start_logit).tolist()[::-1][:n_best]
            end_indexes = np.argsort(end_logit).tolist()[::-1][:n_best]
        
    
            for start_index in start_indexes:
                for end_index in end_indexes:
                
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                       ):
                        continue

                    answers.append({
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                        })

    
            # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})
    
    metric = evaluate.load("squad")

    theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    
    metric_ = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    return predicted_answers,metric_


pred_answers,metrics_ = predict_answers_and_evaluate(start_logits,end_logits,eval_set,data_val_sample)
metrics_

from datasets import load_dataset
dataset = load_dataset("squad")

#lets sample a small dataset
dataset['train'] = dataset['train'].select([i for i in range(1000)])
dataset['validation'] = dataset['validation'].select([i for i in range(250)])

dataset

from torch.utils.data import DataLoader, Dataset


class DataQA(Dataset):
    def __init__(self, dataset,mode="train"):
        self.mode = mode
        
        
        if self.mode == "train":
            # sampling
            self.dataset = dataset["train"]
            self.data = self.dataset.map(train_data_preprocess,
                                                      batched=True,
                            remove_columns= dataset["train"].column_names)
        
        else:
            self.dataset = dataset["validation"]
            self.data = self.dataset.map(preprocess_validation_examples,
            batched=True,remove_columns = dataset["validation"].column_names,
               )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        out = {}
        example = self.data[idx]
        out['input_ids'] = torch.tensor(example['input_ids'])
        out['attention_mask'] = torch.tensor(example['attention_mask'])

        
        if self.mode == "train":

            out['start_positions'] = torch.unsqueeze(torch.tensor(example['start_positions']),dim=0)
            out['end_positions'] = torch.unsqueeze(torch.tensor(example['end_positions']),dim=0)
            
        return out
        
trained_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)


train_dataset = DataQA(dataset,mode="train")
val_dataset = DataQA(dataset,mode="validation")



for i,d in enumerate(train_dataset):
    for k in d.keys():
        print(k + ' : ', d[k].shape)
    print('--'*40)

    if i == 3:
        break
        
print('__'*50)

for i,d in enumerate(val_dataset):
    for k in d.keys():
        print(k + ' : ', len(d[k]))
    print('--'*40)
    
    if i == 3:
        break
    
    from transformers import default_data_collator
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=2,
)
eval_dataloader = DataLoader(
    val_dataset, collate_fn=default_data_collator, batch_size=2
)




for batch in train_dataloader:
   print(batch['input_ids'].shape)
   print(batch['attention_mask'].shape)
   print(batch['start_positions'].shape)
   print(batch['end_positions'].shape)
   break

print('---'*20)

for batch in eval_dataloader:
   print(batch['input_ids'].shape)
   print(batch['attention_mask'].shape)
   break

from transformers import DistilBertForQuestionAnswering
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Available device: {device}')

checkpoint =  "distilbert-base-uncased"
model = DistilBertForQuestionAnswering.from_pretrained(checkpoint)
model = model.to(device)

from transformers import AdamW
from tqdm.notebook import tqdm
import datetime
import numpy as np
import collections
import evaluate

optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 5

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
print(total_steps)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# we need processed validation data to get offsets at the time of evaluation
validation_processed_dataset = dataset["validation"].map(preprocess_validation_examples,
            batched=True,remove_columns = dataset["validation"].column_names,
               )


import random,time
import numpy as np

# to reproduce results
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


#storing all training and validation stats
stats = []


#to measure total training time
total_train_time_start = time.time()

for epoch in range(epochs):
    print(' ')
    print(f'=====Epoch {epoch + 1}=====')
    print('Training....')
     
    # ===============================
    #    Train
    # ===============================   
    # measure how long training epoch takes
    t0 = time.time()
     
    training_loss = 0
    # loop through train data
    model.train()
    for step,batch in enumerate(train_dataloader):
         
        # we will print train time in every 40 epochs
        if step%40 == 0 and not step == 0:
              elapsed_time = format_time(time.time() - t0)
              # Report progress.
              print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed_time))

         
       
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
            


        #set gradients to zero
        model.zero_grad()

        result = model(input_ids = input_ids, 
                        attention_mask = attention_mask,
                        start_positions = start_positions,
                        end_positions = end_positions,
                        return_dict=True)
         
        loss = result.loss
    
        #accumulate the loss over batches so that we can calculate avg loss at the end
        training_loss += loss.item()      

        #perform backward prorpogation
        loss.backward()

        # update the gradients
        optimizer.step()

    # calculate avg loss
    avg_train_loss = training_loss/len(train_dataloader) 
 
    # calculates training time
    training_time = format_time(time.time() - t0)
     
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    
    
    # ===============================
    #    Validation
    # ===============================
     
    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()
     

    start_logits,end_logits = [],[]
    for step,batch in enumerate(eval_dataloader):
         
       
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

         
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():  
             result = model(input_ids = input_ids, 
                        attention_mask = attention_mask,return_dict=True)
        


        start_logits.append(result.start_logits.cpu().numpy())
        end_logits.append(result.end_logits.cpu().numpy())
   

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    # start_logits = start_logits[: len(val_dataset)]
    # end_logits = end_logits[: len(val_dataset)]




    # calculating metrics
    answers,metrics_ = predict_answers_and_evaluate(start_logits,end_logits,validation_processed_dataset,dataset["validation"])
    print(f'Exact match: {metrics_["exact_match"]}, F1 score: {metrics_["f1"]}')


    print('')
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    print("  Validation took: {:}".format(validation_time))

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_train_time_start)))
