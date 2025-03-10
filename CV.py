import re
import gc
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import time
import json
import torch
import tarfile
import random
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaConfig,
)

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

print("Kfold with NO Upsampling")
BATCH_SIZE = 8
EPOCHS = 20
DROPUT_RATE = 0.5147110342609421
UPSAMPLE = False
DOWNSAMPLE = False
ALPHA = 0.5255022533422351
GAMMA = 1.7175764384007821
EPS = 8.105701264441974e-05
WDECAY = 0.08407571894449765
NUM_WARMUP = 24
LR = 9.274425360018714e-06
RUN_ID = "".join(random.choice("0123456789ABCDEF") for i in range(6))
FILE = "../Data/"
TARGET_CLASS = "class_fraud"  # "class_freefair" or "class_fraud"
SAVE_MODEL = True

print("Stratified KFold with upsampling")
print(f"RUN ID: {RUN_ID}")
print(f"Data Used: {FILE}")
print(f"Target Class: {TARGET_CLASS}")
print(
    f"Up Sample: {UPSAMPLE}, Down Sample: {DOWNSAMPLE}"
)
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"Dropout Rate: {DROPUT_RATE}")
print(f"Alpha: {ALPHA}, Gamma: {GAMMA}")
print(f"Learning Rate: {LR}, Epsilon: {EPS}, Weight Decay: {WDECAY}")
print(f"Num Warmup: {NUM_WARMUP}")

# Fixes Out of Memory Issues
gc.collect()
torch.cuda.empty_cache()


def df_to_tensor(X, y):
    input_ids, attention_masks = encode_texts(
        X["text"].values, tokenizer, truncate_middle=False
    )
    labels = torch.tensor(y.values).long()
    return TensorDataset(input_ids, attention_masks, labels)


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def check_classes(df, column):
    print(df[column].value_counts(dropna=False) / df.shape[0] * 100, "\n")
    print(df[column].value_counts())


def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df, min_wc=5):

    df["text"] = df["text"].astype(str).apply(clean_text)

    df["party"] = df["party"].astype(str).apply(lambda x: (x + " ") * 5)
    df["text"] = df["party"] + " " + df["text"]

    df["wc"] = df["text"].apply(lambda x: len(x.split()))
    df = df[df["wc"] > min_wc]

    return df


class RobertaEI(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaEI, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(
            p=DROPUT_RATE
        )  # You can adjust the dropout probability
        # self.classifier = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(p=DROPUT_RATE),
        #     nn.Linear(config.hidden_size, config.num_labels),
        #     # nn.Linear(config.hidden_size, config.hidden_size),
        #     # Add one more fully connected layer
        #     # nn.ReLU(),
        #     # nn.Dropout(p=DROPUT_RATE),
        #     # nn.Linear(config.hidden_size, config.num_labels),
        # )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss_fct = FocalLoss(alpha=ALPHA, gamma=GAMMA)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for Class 1
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


le = LabelEncoder()
df = pd.DataFrame()
if os.path.isdir(FILE):
    files = os.listdir(FILE)
    print("List of files: ", files)
    for file in files:
        if file.endswith(".xlsx"):
            # Only read the following columuns: ['Author Name', 'Message', 'Party', 'Free Fair', 'Fraud']
            check_cols = ["Author Name", "Message", "Party", "Free Fair", "Fraud"]
            file_df = pd.read_excel(FILE + file, usecols=lambda x: x in check_cols)
            print("Loaded File: ", file)
            # Rename Author_name to page_name, Message to text, Party to party, Free Fair to class_freefair, Fraud to class_fraud
            file_df.rename(
                columns={
                    "Author Name": "page_name",
                    "Message": "text",
                    "Party": "party",
                    "Free Fair": "class_freefair",
                    "Fraud": "class_fraud",
                },
                inplace=True,
            )

            # Convert class_ columns to integers, text to string
            file_df["class_freefair"] = file_df["class_freefair"].fillna(0).astype(int)
            file_df["class_fraud"] = file_df["class_fraud"].fillna(0).astype(int)
            file_df["class_fraud"] = file_df["class_fraud"].replace(9, 1)
            file_df["text"] = file_df["text"].fillna("").astype(str)

            # party should only be Democrat or Republican, check if these strings are present in the column and assign the value accordingly
            # Check if party column is present in the dataframe
            if "party" in file_df.columns:
                file_df["party"] = (
                    file_df["party"]
                    .astype(str)
                    .apply(
                        lambda x: (
                            "Democrat"
                            if "Democrat" in x
                            else "Republican" if "Republican" in x else "Other"
                        )
                    )
                )

            # Concat to df
            df = pd.concat([df, file_df], ignore_index=True)

df = preprocess(df)


print("Main Class Distribution")
check_classes(df, TARGET_CLASS)

df[TARGET_CLASS] = le.fit_transform(df[TARGET_CLASS])


X, y = df["text"].to_frame(), df[TARGET_CLASS].to_frame()
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=28
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=28
)

# if UPSAMPLE:
#     ros = RandomOverSampler(random_state=28)
#     X_train, y_train = ros.fit_resample(X_train, y_train)

#     print("\nAfter Upsampling:")
#     check_classes(y_train, TARGET_CLASS)

# if DOWNSAMPLE:
#     rus = RandomUnderSampler(random_state=28)
#     X_train, y_train = rus.fit_resample(X_train, y_train)

#     print("\nAfter Downsampling:")
#     check_classes(y_train, TARGET_CLASS)

print("Y_train Class Distribution")
check_classes(y_train, TARGET_CLASS)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# For Weighted Loss Function
# class_counts = np.bincount(y_train["class_freefair"])
# total_samples = y_train.shape[0]
# class_weights = total_samples / (len(class_counts) * class_counts)
# print("Class Weights: ", class_weights)
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Classess
classList = [f"Class {x}" for x in range(len(y_train.value_counts()))]


def encode_texts(texts, tokenizer, truncate_middle=True):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Truncate middle (First 129 and Last 382 tokens)
        if truncate_middle:
            input_id = torch.cat(
                (
                    encoded_dict["input_ids"][:, :129],
                    encoded_dict["input_ids"][:, -383:],
                ),
                dim=1,
            )
            attention_mask = torch.cat(
                (
                    encoded_dict["attention_mask"][:, :129],
                    encoded_dict["attention_mask"][:, -383:],
                ),
                dim=1,
            )
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
        else:
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        b_input_ids, b_attention_masks, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_attention_masks = b_attention_masks.to(device)
        b_labels = b_labels.to(device)

        model.zero_grad()

        outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, device, fold=None, calc_metrics=False):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        b_input_ids, b_attention_masks, b_labels = batch
        b_input_ids = b_input_ids.to(device)
        b_attention_masks = b_attention_masks.to(device)
        b_labels = b_labels.to(device)

        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attention_masks)

        logits = outputs[0]
        predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(b_labels.cpu().numpy())

    if calc_metrics:
        print("Metrics for Model Fold: ", fold)
        print(classification_report(true_labels, predictions, target_names=classList, labels=[0, 1]))
        conf_matrix = confusion_matrix(true_labels, predictions)
        print(conf_matrix)
        # plt.figure(figsize=(10, 7))
        # sns.heatmap(
        #     conf_matrix,
        #     annot=True,
        #     fmt="d",
        #     cmap="Blues",
        #     xticklabels=classList,
        #     yticklabels=classList,
        # )
        # plt.xlabel("Predicted")
        # plt.ylabel("True")
        # plt.title("Confusion Matrix")
        # plt.savefig(f"./Graphs/confmatrix_{RUN_ID}_Fold_{fold}.png")

    return f1_score(true_labels, predictions, average="weighted")


# Encode training dataset
train_data = df_to_tensor(X_train, y_train)
train_dataloader = DataLoader(
    train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE
)

# Not needed for CV: Encode validation dataset
# val_data = df_to_tensor(X_val, y_val)
# validation_dataloader = DataLoader(
#     val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE
# )

labels = torch.tensor(y_train[TARGET_CLASS].values).long()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=28)
# kf = KFold(n_splits=5)

# for fold_n, (train_index, val_index) in enumerate(kf.split(X_train, y_train)):
#     print(f"\nFold {fold_n}:")
#     print("Validation class distribution:", np.bincount(y_train.iloc[val_index][TARGET_CLASS]))
    
val_f1_scores = []
train_f1_scores = []

best_model = None
for train_index, val_index in kf.split(X_train, y_train):
# for train_index, val_index in kf.split(X_train):
    fold_n = len(val_f1_scores)
    train_texts, val_texts = (
        X_train["text"].iloc[train_index],
        X_train["text"].iloc[val_index],
    )
    train_labels, val_labels = labels[train_index], labels[val_index]

    if UPSAMPLE:
        ros = RandomOverSampler(random_state=28)
        # Create a dummy feature index for resampling since we only need to upsample train_texts and train_labels.
        dummy_idx = np.arange(len(train_texts)).reshape(-1, 1)
        resampled_idx, resampled_labels = ros.fit_resample(dummy_idx, train_labels.numpy())
        resampled_idx = resampled_idx.flatten()
        train_texts = train_texts.iloc[resampled_idx]
        train_labels = torch.tensor(resampled_labels, dtype=torch.long)

    train_input_ids, train_attention_masks = encode_texts(train_texts, tokenizer)
    val_input_ids, val_attention_masks = encode_texts(val_texts, tokenizer)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE
    )
    val_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=BATCH_SIZE
    )

    # New model defined again for each fold
    model = RobertaEI.from_pretrained(
        "roberta-base", num_labels=len(y_train.value_counts())
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR, eps=EPS, weight_decay=WDECAY)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP, num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, scheduler, device)

    train_f1 = evaluate(model, train_dataloader, device, fold_n)
    val_f1 = evaluate(model, val_dataloader, device, fold_n, calc_metrics=True)

    
    if not val_f1_scores or val_f1 > max(val_f1_scores):
        print(f"Best Val F1 so far is {fold_n}: ", val_f1)
        best_model = model

    val_f1_scores.append(val_f1)
    train_f1_scores.append(train_f1)

    print(f"\nFold {fold_n}: Training F1: {train_f1}, Validation F1: {val_f1}")

print(f"\nAverage Training F1 Score: {np.mean(train_f1_scores)}")
print(f"Average Validation F1 Score: {np.mean(val_f1_scores)}")

print("\nRunning Evaluation on Test Set for the Best Model...")
test_data = df_to_tensor(X_test, y_test)
test_dataloader = DataLoader(
    test_data, sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE
)

best_model.eval()
predictions, true_labels = [], []

for batch in test_dataloader:
    b_input_ids, b_attention_masks, b_labels = batch
    b_input_ids = b_input_ids.to(device)
    b_attention_masks = b_attention_masks.to(device)
    b_labels = b_labels.to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_attention_masks)

    logits = outputs[0]
    predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
    true_labels.extend(b_labels.cpu().numpy())

print(classification_report(true_labels, predictions, target_names=classList, labels=[0, 1]))
conf_matrix = confusion_matrix(true_labels, predictions)
print(conf_matrix)

# Save the best model
if SAVE_MODEL:
    config = best_model.roberta.config  # Get the BERT configuration

    # Define file paths
    bin_file_path = "../Models/robertaei.bin"
    config_file_path = "../Models/robertaei.json"

    tar_file_path = f"../Models/roberta_{TARGET_CLASS}_{RUN_ID}.tar.gz"

    torch.save(model.state_dict(), bin_file_path)

    with open(config_file_path, "w") as f:
        f.write(config.to_json_string())

    # Create tar.gz archive
    with tarfile.open(tar_file_path, "w:gz") as tar:
        tar.add(bin_file_path, arcname="model.bin")
        tar.add(config_file_path, arcname="config.json")

    # Clean up
    os.remove(bin_file_path)
    os.remove(config_file_path)

    print(f"Model and configuration have been saved to {tar_file_path}")
