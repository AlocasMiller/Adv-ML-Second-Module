import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import logging
import argparse
import warnings
import wandb

# Настройка логирования
logging.basicConfig(filename='./data/log_file.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Функция для записи предупреждений в файл логов
def warn_with_log(message, category, filename, lineno, file=None, line=None):
    log = logging.getLogger(__name__)
    log.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

# Перенаправление предупреждений в файл логов
warnings.showwarning = warn_with_log
warnings.filterwarnings('always')  # Всегда выводить предупреждения

class My_TextClassifier_Model:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train(self, train_dataset):
        try:
            device = torch.device("cpu")
            
            # Data loading
            train_data = pd.read_csv(train_dataset)
            
            # Data preprocessing
            train_data['text'] = train_data['text'].apply(lambda x: x.lower())
            
            # Data split
            X_train, X_val, y_train, y_val = train_test_split(train_data['text'], train_data['generated'], test_size=0.2, random_state=42)
    
            # Creating BERT model
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding=True, truncation=True, max_length=128)
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
            
            # Preparing input data for BERT
            encoded_train = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
            encoded_val = tokenizer(X_val.tolist(), padding=True, truncation=True, return_tensors='pt')
            
            # Convert labels to tensors
            train_labels = torch.tensor(y_train.values)
            val_labels = torch.tensor(y_val.values)
            
            # Create TensorDatasets
            train_dataset = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], train_labels)
            val_dataset = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], val_labels)
            
            # DataLoader for efficient processing
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
            
            # Define optimizer and learning rate scheduler
            optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
            epochs = 2

            wandb.init(project="my-text-classifier", name="bert-model")
            wandb.watch(model, log="all")

            for epoch in range(epochs):
                model.train()
                total_loss = 0

                for batch in train_loader:
                    input_ids, attention_mask, labels = batch
                    input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping to avoid exploding gradients
                    optimizer.step()

                avg_train_loss = total_loss / len(train_loader)
                logger.info(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.2f}")
                wandb.log({"Average Training Loss": avg_train_loss})

            model_dir = './data/model/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            torch.save(model.state_dict(), './data/model/model.pth')

            self.logger.info("Model was trained and artifacts were saved at ./model/")
        except Exception as e:
            self.logger.error(f"Error while training: {str(e)}")

    def predict(self, test_dataset):
        try:
            device = torch.device("cpu")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, padding=True, truncation=True, max_length=128)
            
            # Data loading
            test_data = pd.read_csv(test_dataset)
            test_data['text'] = test_data['text'].apply(lambda x: x.lower())

            # Preparing input data for BERT
            test_inputs = tokenizer(test_data['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

            # Moving the input tensor to the same device as the model
            test_inputs = {key: value.to(device) for key, value in test_inputs.items()}
            
            # Model loading
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            model.load_state_dict(torch.load('./data/model/model.pth'))
            model = model.to(device)
            model.eval()  # Перевод модели в режим оценки

            # Generating predictions using the trained model
            with torch.no_grad():
                outputs = model(**test_inputs)
                logits = outputs.logits 

            # Assuming the first logits column corresponds to the negative class (not AI generated),
            # and the second column corresponds to the positive class (AI generated)
            predictions = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # Creating a Submission DataFrame with Essay ID and Corresponding Predictions
            submission = pd.DataFrame({
                'id': test_data['id'],
                'generated': predictions
            })

            # Saving a DataFrame to send to a CSV file
            submission.to_csv('./data/results.csv', index=False)

            self.logger.info(f"Predictions was succesfully saved results.csv")
        except Exception as e:
            self.logger.error(f"Error while predicting: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Обучение и предсказание с использованием модели")
    parser.add_argument("action", type=str, choices=["train", "predict"], help="Выберите действие: обучение (train) или предсказание (predict)")
    parser.add_argument("--dataset", type=str, help="Путь к файлу набора данных")
    parser.add_argument("--text", type=str, help="Текст для предсказания")
    args = parser.parse_args()

    model = My_TextClassifier_Model()

    if args.action == "train":
        model.train(args.dataset)
    elif args.action == "predict":
        prediction = model.predict(args.dataset)
        print(f"Predicted probability: {prediction}")

if __name__ == "__main__":
    main()
    print("It's worked!!!")