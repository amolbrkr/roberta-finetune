import os
import time
import datetime
import random
import gc
import re
import torch
import optuna
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from transformers import (
    RobertaModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

DATA_DIR = "../Data/"
TARGET_CLASS = "class_freefair"


def df_to_tensor(X, y, tokenizer):
    input_ids, attention_masks = encode_texts(
        X["text"].values, tokenizer, truncate_middle=False
    )
    labels = torch.tensor(y.values).long()
    return TensorDataset(input_ids, attention_masks, labels)


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
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub(r"[\r\n]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess(df, min_wc=5):

    df["text"] = df["text"].astype(str).apply(clean_text)

    df["party"] = df["party"].astype(str).apply(lambda x: (x + " ") * 5)
    df["text"] = df["party"] + " " + df["text"]

    df["wc"] = df["text"].apply(lambda x: len(x.split()))
    df = df[df["wc"] > min_wc]

    return df


class RobertaEI(RobertaForSequenceClassification):
    def __init__(self, config, dropout_rate=0.1, alpha=0.25, gamma=2.0):
        super(RobertaEI, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(
            p=dropout_rate
        )  # You can adjust the dropout probability
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.alpha = alpha
        self.gamma = gamma

        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
            loss_fct = FocalLoss(alpha=self.alpha, gamma=self.gamma)
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


def load_data(DATA_DIR):
    df = pd.DataFrame()
    if os.path.isdir(DATA_DIR):
        files = os.listdir(DATA_DIR)
        print("List of files: ", files)
        for file in files:
            if file.endswith(".xlsx"):
                # Only read the following columuns: ['Author Name', 'Message', 'Party', 'Free Fair', 'Fraud']
                check_cols = ["Author Name", "Message", "Party", "Free Fair", "Fraud"]
                file_df = pd.read_excel(
                    DATA_DIR + file, usecols=lambda x: x in check_cols
                )
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
                file_df["class_freefair"] = (
                    file_df["class_freefair"].fillna(0).astype(int)
                )
                file_df["class_fraud"] = file_df["class_fraud"].fillna(0).astype(int)
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
    return df


# Main objective function for hyperparameter search
def objective(trial):
    # Set random seed for reproducibility
    random.seed(28)
    np.random.seed(28)
    torch.manual_seed(28)

    df = load_data(DATA_DIR)
    df = preprocess(df)

    # Label Encoding
    le = LabelEncoder()
    df[TARGET_CLASS] = le.fit_transform(df[TARGET_CLASS])
    check_classes(df, TARGET_CLASS)

    # Split dataset
    X = df["text"].to_frame()
    y = df[TARGET_CLASS].to_frame()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=28
    )

    # Hyperparameters suggestions for batch size and learning rate
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_float("lr", 8e-6, 2e-5, log=True)
    num_epochs = trial.suggest_categorical("num_epochs", [5, 8, 10, 12, 15, 20])

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: ", device)

    # For Weighted Loss Function
    # _labels = y_train[TARGET_CLASS]
    # _classes = np.unique(_labels)
    # class_weights = compute_class_weight(
    # class_weight="balanced", classes=_classes, y=_labels
    # )
    # print("Calc. Class Weights: ", class_weights)
    # Convert class_weights to tensor
    # class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Tokenizer and dataset creation
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = df_to_tensor(X_train, y_train, tokenizer)
    val_dataset = df_to_tensor(X_val, y_val, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    val_dataloader = DataLoader(
        val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size
    )

    # Model setup
    # config = RobertaConfig.from_pretrained("roberta-base", num_labels=len(np.unique(df[TARGET_CLASS])))
    model = RobertaEI.from_pretrained(
        "roberta-base",
        num_labels=len(np.unique(df[TARGET_CLASS])),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.6),
        alpha=trial.suggest_float("alpha", 0.1, 0.9),
        gamma=trial.suggest_float("gamma", 1.0, 5.0),
    )
    model.to(device)

    # Optimizer and learning rate scheduler (using minimal warmup steps for trial)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        eps=trial.suggest_float("eps", 1e-7, 1e-4),
        weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
    )
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=trial.suggest_int("warmup_steps", 0, 50),
        num_training_steps=total_steps,
    )

    # Training for 1 epoch
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(
                input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels
            )
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        # Optional: Print loss for the current epoch
        print(
            f"Epoch {epoch+1}/{num_epochs} Train Loss: {total_train_loss/len(train_dataloader)}"
        )

    # Validation
    model.eval()
    total_preds = []
    total_true = []
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids=b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        total_preds.extend(preds)
        total_true.extend(b_labels.detach().cpu().numpy())

    # Print classification report
    print("Classification Report: ")
    print(classification_report(total_true, total_preds))

    # Use weighted F1 score as objective metric
    f1 = f1_score(total_true, total_preds, average="weighted")
    print("Validation F1 Score: ", f1)

    # return f1
    # Instead of returning the F1 Score, return the F1 score for postive class (class 1)
    # This is done to ensure that the model is not biased towards the majority class
    f1_positive = f1_score(total_true, total_preds, average="binary", pos_label=1)
    print("Validation F1 Score (Positive Class): ", f1_positive)
    return f1_positive


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="RobertaEI_2",
        direction="maximize",
        storage="sqlite:///hyperparamsearch.db",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)
    print("Best trial:")
    trial = study.best_trial
    print("  F1 Score: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
