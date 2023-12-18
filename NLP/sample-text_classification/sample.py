# %%
from datasets import load_dataset

dataset = load_dataset("tyqiangz/multilingual-sentiments", "japanese")


# %%
from transformers import AutoTokenizer

model_ckpt = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


# %%
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

# %%
import pandas as pd

sample_encoded = dataset_encoded["train"][0]
pd.DataFrame(
    [
        sample_encoded["input_ids"],
        sample_encoded["attention_mask"],
        tokenizer.convert_ids_to_tokens(sample_encoded["input_ids"]),
    ],
    ["input_ids", "attention_mask", "tokens"],
).T


# %%
import torch
from transformers import AutoModelForSequenceClassification

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_labels = 3

model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)


# %%
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


# %%
from transformers import TrainingArguments

batch_size = 16
logging_steps = len(dataset_encoded["train"]) // batch_size
model_name = "sample-text-classification-bert"

training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    disable_tqdm=False,
    logging_steps=logging_steps,
    push_to_hub=False,
    log_level="error",
)

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset_encoded["train"],
    eval_dataset=dataset_encoded["validation"],
    tokenizer=tokenizer,
)
trainer.train()

# %%
preds_output = trainer.predict(dataset_encoded["validation"])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_preds = np.argmax(preds_output.predictions, axis=1)
y_valid = np.array(dataset_encoded["validation"]["label"])
labels = dataset_encoded["train"].features["label"].names


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


plot_confusion_matrix(y_preds, y_valid, labels)
