#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install --upgrade pip


# In[2]:


pip install datasets


# In[3]:


pip install dataset


# In[4]:


pip install transformers


# In[5]:


pip install evaluate


# In[6]:


pip install torch


# In[7]:


get_ipython().system('python --version')


# In[2]:


import pandas as pd
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
#import evaluate
from transformers import TrainingArguments, Trainer
from datasets import load_dataset


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",cache="")


# In[9]:


get_ipython().system('ls')


# In[10]:


dataset = load_dataset('csv', data_files={'train': '/gpfs/fs1/home/j/jcaunedo/umar1/argument/data_subset_processed.csv'})


# In[11]:


dataset


# In[12]:


train_dataset, validation_dataset = dataset['train'].train_test_split(test_size=0.3).values()


# In[13]:


dd = DatasetDict({"train":train_dataset,"validation":validation_dataset})


# In[14]:


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
  return tokenizer(examples["statement"], examples["argument"], padding="max_length", truncation=True)


# In[15]:


tokenized_datasets = dd.map(tokenize_function, batched=True)


# In[16]:


tokenized_datasets


# In[17]:


small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(1000))


# In[18]:


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=10)


# In[19]:


training_args = TrainingArguments(output_dir="test_trainer")


# In[26]:


from datasets import load_metric
metric = load_metric("accuracy")


# In[27]:


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# In[22]:


args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01
)


# In[23]:


from transformers import default_data_collator
data_collator = default_data_collator


# In[24]:


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# In[ ]:


trainer.train()


# In[ ]:





# In[ ]:




