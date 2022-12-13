import pandas as pd
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from transformers import TrainerCallback

dataset = load_dataset('csv', data_files={'train': '/home/j/jcaunedo/umar1/argument/data_train.csv','test': '/home/j/jcaunedo/umar1/argument/data_test.csv'},cache_dir ="/scratch/j/jcaunedo/umar1/argument/cache")
print("Dataset loaded")
train_dataset, test_dataset = dataset['train'],dataset['test']

dd = DatasetDict({"train":train_dataset,"test":test_dataset})

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",cache_dir="/scratch/j/jcaunedo/umar1/argument/cache")

def tokenize_function(examples):
  return tokenizer(examples["input"],  padding="max_length", truncation=True)

tokenized_datasets = dd.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)#.select(range(300))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42)#.select(range(50))

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=6,cache_dir="/scratch/j/jcaunedo/umar1/argument/cache",use_cache=False)
model.config.hidden_dropout_prob = 0.5
model.config.attention_probs_dropout_prob = 0.5
print("Model loaded")
from datasets import load_metric


print("Training argument started")

args = TrainingArguments(
    output_dir="/scratch/j/jcaunedo/umar1/bert_nd/",
    evaluation_strategy = "steps",
    eval_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    fp16=True,
)

print("Training argument")

from transformers import default_data_collator
data_collator = default_data_collator

metric = load_metric("accuracy.py",cache_dir="/scratch/j/jcaunedo/umar1/argument/cache")
#print("metric load")

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    #accuracy = np.mean(predictions==labels)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
print("Training started")
trainer.add_callback(CustomCallback(trainer))
print(trainer.evaluate(small_train_dataset))
print(trainer.evaluate(small_eval_dataset))