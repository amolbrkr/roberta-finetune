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

# import nlpaug.augmenter.word as naw

from torch import nn
from sklearn.model_selection import train_test_split, KFold
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

BATCH_SIZE = 32
EPOCHS = 12
UPSAMPLE = True
DOWNSAMPLE = False
DROPUT_RATE = 0.583
ALPHA = 0.338
GAMMA = 1.30
LR = 0.000014523474567726823
EPS = 9.730e-7
WDECAY = 0.03983034553582014
NUM_WARMUP = 37
RUN_ID = "".join(random.choice("0123456789ABCDEF") for i in range(6))
FILE = "./Data/"
FREEZE_LAYERS = False
USE_CLASS_WEIGHTS = True
TARGET_CLASS = "class_freefair"

print(f"RUN ID: {RUN_ID}")
print(f"Data Used: {FILE}")
print(f"Target Class: {TARGET_CLASS}")
print(
    f"Freezed Layers: {FREEZE_LAYERS}, Up Sample: {UPSAMPLE}, Down Sample: {DOWNSAMPLE}"
)
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"Class Weights: {USE_CLASS_WEIGHTS}")
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
        self.dropout = nn.Dropout(p=DROPUT_RATE)  # You can adjust the dropout probability
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
            loss_fct = FocalLoss(alpha = ALPHA, gamma = GAMMA)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # Weight for Class 1
        self.gamma = gamma  # Focusing parameter

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


le = LabelEncoder()
# df = pd.read_csv(FILE)

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
print(df[TARGET_CLASS].value_counts())

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

if UPSAMPLE:
    ros = RandomOverSampler(random_state=28)
    X_train, y_train = ros.fit_resample(X_train, y_train)

if DOWNSAMPLE:
    rus = RandomUnderSampler(random_state=28)
    X_train, y_train = rus.fit_resample(X_train, y_train)

print(X_train.shape, X_val.shape, X_test.shape)

print("Y_train Class Distribution")
check_classes(y_train, TARGET_CLASS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
# config.num_labels = len(y_train.value_counts())
# tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
# model = ModEI(config)
# model = AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", config=config)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaEI.from_pretrained(
    "roberta-base", num_labels=len(y_train.value_counts())
)
model.to(device)



# For Weighted Loss Function
_labels = y_train[TARGET_CLASS]
_classes = np.unique(_labels)
class_weights = compute_class_weight(
    class_weight="balanced", classes=_classes, y=_labels
)
print("Calc. Class Weights: ", class_weights)
# Convert class_weights to tensor
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Freeze layers in the Model
if FREEZE_LAYERS:
    for name, param in model.named_parameters():
        if (
            "encoder.layer.10" not in name
            and "encoder.layer.11" not in name
            and "pooler" not in name
            and "classifier" not in name
        ):
            param.requires_grad = False

    # Print to verify
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")


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


# Encode training dataset
train_data = df_to_tensor(X_train, y_train)
train_dataloader = DataLoader(
    train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE
)

# Encode validation dataset
val_data = df_to_tensor(X_val, y_val)
validation_dataloader = DataLoader(
    val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE
)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    eps=EPS,
    weight_decay=WDECAY,
)

total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=NUM_WARMUP, num_training_steps=total_steps
)

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, EPOCHS):
    print("")
    print("======== Epoch {:} / {:} ========".format(epoch_i + 1, EPOCHS))
    print("Training...")

    t0 = time.time()
    total_train_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print(
                "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                    step, len(train_dataloader), elapsed
                )
            )

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()

        output = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )

        # loss = loss_fn(output.logits, b_labels.squeeze())
        loss = output[0]
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            output = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )

        loss = output[0]
        logits = output[1]
        # loss = loss_fn(output.logits, b_labels.squeeze())
        total_eval_loss += loss.item()
        label_ids = b_labels.to("cpu").numpy()
        total_eval_accuracy += f1_score(
            torch.argmax(logits, dim=1).cpu().numpy(),
            label_ids,
            average="weighted",
        )

    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  F1 Score: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    validation_time = format_time(time.time() - t0)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            "epoch": epoch_i + 1,
            "Training Loss": avg_train_loss,
            "Valid. Loss": avg_val_loss,
            "Valid. Accur.": avg_val_accuracy,
            "Training Time": training_time,
            "Validation Time": validation_time,
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


df_stats = pd.DataFrame(data=training_stats).set_index("epoch")
sns.set(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.plot(df_stats["Training Loss"], "b-o", label="Training Loss")
plt.plot(df_stats["Valid. Loss"], "g-o", label="Validation Loss")
plt.plot(df_stats["Valid. Accur."], "r-o", label="Validation F1")
plt.title("Training Stats")
plt.xlabel("Epoch")
plt.legend()
plt.xticks([i for i in range(1, EPOCHS + 1)])
# plt.show()
plt.savefig(f"./Graphs/training_stats_{RUN_ID}.png")

print("Running Evaluaction on Test Set...")
test_data = df_to_tensor(X_test, y_test)
test_dataloader = DataLoader(
    test_data, sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE
)

t0 = time.time()
model.eval()
total_eval_accuracy = 0
total_eval_loss = 0

for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
        output = model(
            b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            labels=b_labels,
        )

    loss, logits = output[0], output[1]
    total_eval_loss += loss.item()
    label_ids = b_labels.to("cpu").numpy()
    total_eval_accuracy += f1_score(
        torch.argmax(logits, dim=1).cpu().numpy(), label_ids, average="weighted"
    )

print("Total Test Loss: ", total_eval_loss / len(test_dataloader))
print("Test F1 Score: ", total_eval_accuracy / len(test_dataloader))


def get_predictions(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        # logits = outputs.logits
        logits = outputs[0]
        predictions.append(logits.detach().cpu().numpy())
        true_labels.append(b_labels.detach().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()

    return preds_flat, labels_flat


preds_flat, labels_flat = get_predictions(model, test_dataloader)

classList = [f"Class {x}" for x in range(len(y_train.value_counts()))]
print("Classification Report:")
print(classification_report(labels_flat, preds_flat, target_names=classList))
conf_matrix = confusion_matrix(labels_flat, preds_flat)
print("Confusion Matrix:")
print(conf_matrix)
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classList,
    yticklabels=classList,
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
# plt.show()
plt.savefig(f"./Graphs/confmatrix_{RUN_ID}.png")


# torch.save(model.state_dict(), "../Models/partbert.pt")
