import numpy as np
import pandas as pd
from pathlib import Path
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, trainer
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from transformers import EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

np.random.seed(42)

test_df = pd.read_csv('datasets/test_essays.csv')
submission_df = pd.read_csv('datasets/sample_submission.csv')
train_df = pd.read_csv("datasets/train_v2_drcat_02.csv")
kf_df = pd.read_csv('datasets/kf_df.csv')

kf_df = kf_df.rename(columns={'prompt_title': 'prompt_name'})
kf_df['label'] = 1
kf_df['source'] = 'kf'
kf_df['RDizzl3_seven'] = False

train_df = pd.concat([train_df, kf_df[train_df.columns].sample(30000, random_state=42)])

# print(train_df)
# Step 1. Text Preprocessing
train_df["words_count"] = train_df["text"].apply(lambda x: len(x.split(" ")))
# print(train_df)
train_df.query("label == 0")["words_count"].mean()
train_df.query("label == 1")["words_count"].mean()

train_df["generated"] = train_df["label"].apply(lambda x: 1.0 if x == 1 else 0.0)
train_df["human"] = train_df["label"].apply(lambda x: 1.0 if x == 0 else 0.0)

# Step 2. Modeling

train, test = train_test_split(train_df, test_size=0.30, random_state=42, shuffle=True, stratify=train_df["label"])

train.groupby("label").count()

test.groupby("label").count()

train.to_csv("train.csv")
test.to_csv("test.csv")

# Model Training


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = ['generated', 'human']
id2label = {idx:label for idx, label in enumerate(LABELS)}
label2id = {label:idx for idx, label in enumerate(LABELS)}

def read_csv_binary(filename):
    data = pd.read_csv(filename)
    texts = data['text'].tolist()
    labels = data[LABELS].values


    return texts, labels

# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
            'roc_auc': roc_auc,
            'accuracy': accuracy}
    return metrics

class LLMDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }

        item['labels'] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.labels)


class ClassificationTrainer():
    def __init__(self,
                 pretrained_transformer_name='distilbert-base-cased',
                 dataset_dct={'train':'train.csv', 'test':'test.csv', 'val':'val.csv'},
                 warmup_steps=500,
                 num_train_epochs=3):


        max_samples = {
            'train': 100000,
            'val': 100000,
            'test': 100000,
        }

        train_texts, train_labels = read_csv_binary(dataset_dct['train'])

        if 'test' not in dataset_dct:
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                train_texts, train_labels, test_size=.1)
        else:
            test_texts, test_labels = read_csv_binary(dataset_dct['test'])

        if 'val' not in dataset_dct:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=.1)
        else:
            val_texts, val_labels = read_csv_binary(dataset_dct['val'])

        train_texts = train_texts[:max_samples['train']]
        val_texts = val_texts[:max_samples['val']]
        test_texts = test_texts[:max_samples['test']]

        train_labels = train_labels[:max_samples['train']]
        val_labels = val_labels[:max_samples['val']]
        test_labels = test_labels[:max_samples['test']]

        print(train_texts[0])
        print(train_labels[0])

        self.tokenizer = BertTokenizerFast.from_pretrained(
                pretrained_transformer_name)

#посмотреть по парамкетрам
        train_encodings = self.tokenizer(train_texts, truncation=True, max_length=256, padding=True)
        val_encodings = self.tokenizer(val_texts, truncation=True, max_length=256, padding=True)
        test_encodings = self.tokenizer(test_texts, truncation=True, max_length=256, padding=True)

        self.train_dataset = LLMDDataset(train_encodings, train_labels)
        self.val_dataset = LLMDDataset(val_encodings, val_labels)
        self.test_dataset = LLMDDataset(test_encodings, test_labels)

        self.model = DistilBertForSequenceClassification.from_pretrained(
                pretrained_transformer_name, num_labels=len(LABELS), problem_type="multi_label_classification",  id2label=id2label, label2id=label2id)

        self.metric = {metric:load_metric(metric) for metric in ['f1', 'precision', 'recall', 'accuracy']}

        self.training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=num_train_epochs, # total number of training epochs
            per_device_train_batch_size=
            8,  # batch size per device during training
            per_device_eval_batch_size=8,  # batch size for evaluation
            warmup_steps=
            warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_strategy='epoch',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            save_total_limit = 3,
        )

        self.trainer = Trainer(
            model=self.
            model,  # the instantiated 🤗 Transformers model to be trained
            args=self.training_args,  # training arguments, defined above
            train_dataset=self.train_dataset,  # training dataset
            eval_dataset=self.val_dataset,  # evaluation dataset
            compute_metrics=self.compute_metrics,
        )


    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions,
                tuple) else p.predictions
        result = multi_label_metrics(
            predictions=preds,
            labels=p.label_ids)
        return result


    def inference(self, predict_dataset=None):
        if predict_dataset is None:
            predict_dataset = self.test_dataset
        predictions = self.trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        return predictions

classification_trainer = ClassificationTrainer(
    pretrained_transformer_name='cointegrated/rubert-tiny2',
    dataset_dct={'train':'train.csv', 'test': 'test.csv'},
    warmup_steps=100,
    num_train_epochs=3
)

classification_trainer.trainer.train()

metrics = classification_trainer.trainer.evaluate()

classification_trainer.trainer.log_metrics("after_train_eval", metrics)
classification_trainer.trainer.save_metrics("after_train_eval",
                                            metrics)

text = 'Всем привет!'



encoding = classification_trainer.tokenizer(text, return_tensorse='pt')
encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

outputs = trainer.model(**encoding)


logits = outputs.logits

# apply sigmoid + threshold
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(logits.squeeze().cpu())
predictions = np.zeros(probs.shape)
predictions[np.where(probs >= 0.5)] = 1
# turn predicted id's into actual label names
predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
print(predicted_labels)

def inference(trainer, predict_dataset=None):
    if predict_dataset is None:
        predict_dataset = trainer.test_dataset
    predictions = trainer.trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.argmax(predictions, axis=1)

    return predictions

preds = inference(classification_trainer)
test["pred_label"] = [id2label[x] for x in preds]

results = test[["text", "label", "pred_label"]]

print(LABELS)
results[results["label"] == "basis"].sample(100).to_excel("model_preds_test_basis.xlsx")
results[results["label"] == "reminder"].sample(100).to_excel("model_preds_test_reminder.xlsx")
results[results["label"] == "famil"].sample(100).to_excel("model_preds_test_famil.xlsx")
results[results["label"] == "condition"].sample(100).to_excel("model_preds_test_condition.xlsx")
results[results["label"] == "preambs"].sample(100).to_excel("model_preds_test_preambs.xlsx")
results[results["label"] == "requests"].to_excel("model_preds_test_requests.xlsx")
results[results["label"] == "inf_task"].sample(100).to_excel("model_preds_test_inf_task.xlsx")
results[results["label"] == "requests"]
# Save Trained Model
# !cp -r /content/results $DATA_PATH_CLOUD