# %%
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification

dataset = load_dataset("tyqiangz/multilingual-sentiments", "japanese")
model_ckpt = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_labels = 3
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt, num_labels=num_labels
).to(device)

# %%
# データ削減
for i in ["train", "validation", "test"]:
    dataset[i] = dataset[i].train_test_split(
        test_size=0.9, shuffle=True, seed=0
    )["train"]


# %%
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)


# %%
from sklearn.metrics import accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


# %%
from transformers import TrainingArguments

batch_size = 16
logging_steps = len(dataset_encoded["train"]) // batch_size
model_name = "sample-text-classification-bert"

training_args = TrainingArguments(
    output_dir=model_name,
    num_train_epochs=1,
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
preds_output = trainer.predict(
    dataset_encoded["test"]
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_preds = np.argmax(preds_output.predictions, axis=1)
y_valid = np.array(
    dataset_encoded["test"]["label"]
)
labels = (
    dataset_encoded["test"]
    .features["label"]
    .names
)


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


plot_confusion_matrix(y_preds, y_valid, labels)

# %%
import textattack
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
import gensim
from textattack.shared import GensimWordEmbedding
from textattack.transformations import WordSwapEmbedding
from textattack.search_methods import GreedyWordSwapWIR


model_attack = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
tokenizer_attack = AutoTokenizer.from_pretrained(model_ckpt)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
    trainer.model, tokenizer_attack
)

goal_function = textattack.goal_functions.UntargetedClassification(
    model_wrapper
)

gensim_model = gensim.models.KeyedVectors.load_word2vec_format(
    "../sample-text-attack/model.vec", binary=False
)
word_embedding = GensimWordEmbedding(gensim_model)
transformation = WordSwapEmbedding(
    max_candidates=50, embedding=word_embedding
)
constraints = [
    RepeatModification(),
    StopwordModification(),
    WordEmbeddingDistance(min_cos_sim=0.95),
]
search_method = GreedyWordSwapWIR(wir_method="delete")

# Construct the actual attack
attack = textattack.Attack(
    goal_function, constraints, transformation, search_method
)


# %%
i = 3
input_text = dataset["train"]["text"][i]
label = dataset["train"]["label"][i]  # Positive
attack_result = attack.attack(input_text, label)
print(label)
print(attack_result)

# %%
dataset_attack_test = textattack.datasets.HuggingFaceDataset(
    "tyqiangz/multilingual-sentiments", "japanese", split="test"
)
attack_args = textattack.AttackArgs(
    num_examples=20,
    shuffle=True,
    log_to_csv="log_japanese.csv",
    checkpoint_interval=5,
    checkpoint_dir="checkpoints",
    disable_stdout=True,
)

attacker = textattack.Attacker(attack, dataset_attack_test, attack_args)
attacker.attack_dataset()


# %%
