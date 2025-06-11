#!/usr/bin/env python
from torchmetrics import AUROC
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import rnn
from sklearn.model_selection import train_test_split
from ncps.torch import LTC
from ncps.wirings import AutoNCP
import wandb
import os

 
class RecurrentEncoder(torch.nn.Module):
    def __init__(self, input_shape_1, input_shape_2, input_shape_3, \
        dense_units_1_1=32, dense_units_1_2=32, \
        lstm_units_1_1=32, lstm_units_1_2=32, \
        dense_units_2_1=32, dense_units_2_2=32, \
        lstm_units_2_1=32, lstm_units_2_2=32, \
        dense_units_3_1=128, dense_units_3_2=128, dense_units_3_3=16, \
        merge_dense_units_1=64, merge_dense_units_2=32, \
        incl_clusters=True, lstm_block=nn.LSTM, wirings=False):
        super(RecurrentEncoder, self).__init__()
        self.incl_clusters = incl_clusters

        self.shared_dense_1_1 = torch.nn.Linear(input_shape_1[1], dense_units_1_1)
        self.shared_dense_1_2 = torch.nn.Linear(dense_units_1_1, dense_units_1_2)

        #TODO: Add conditional for wirings with AutoNCP    
        self.lstm_1_1 = lstm_block(input_size=dense_units_1_2, hidden_size=lstm_units_1_1, num_layers=1, batch_first=True)
        self.lstm_1_2 = lstm_block(input_size=lstm_units_1_1, hidden_size=lstm_units_1_2, num_layers=1, batch_first=True)

        if incl_clusters:
            self.shared_dense_2_1 = torch.nn.Linear(input_shape_2[1], dense_units_2_1)
            self.shared_dense_2_2 = torch.nn.Linear(dense_units_2_1, dense_units_2_2)
            self.lstm_2_1 = nn.LSTM(input_size=dense_units_2_2, hidden_size=lstm_units_2_1, num_layers=1, batch_first=True)
            self.lstm_2_2 = nn.LSTM(input_size=lstm_units_2_1, hidden_size=lstm_units_2_2, num_layers=1, batch_first=True)

        self.dense_3_1 = nn.Linear(input_shape_3[0], dense_units_3_1)
        self.dense_3_2 = nn.Linear(dense_units_3_1, dense_units_3_2)
        self.dense_3_3 = nn.Linear(dense_units_3_2, dense_units_3_3)

        total_input_size = lstm_units_1_2 + dense_units_3_3
        total_input_size += lstm_units_2_2 if incl_clusters else 0

        self.merge_branches_1 = torch.nn.Linear(total_input_size, merge_dense_units_1)
        self.merge_branches_2 = torch.nn.Linear(merge_dense_units_1, merge_dense_units_2)
        self.merge_branches_3 = torch.nn.Linear(merge_dense_units_2, 1)


    def forward(self, x1, x3, x2=None):
        out_branch_1 = self.apply_branch_1(x1)
        if self.incl_clusters and x2 is not None:
            out_branch_2 = self.apply_branch_2(x2)
        out_branch_3 = self.apply_branch_3(x3)

        all_features = [out_branch_1]
        if self.incl_clusters and x2 is not None:
            all_features.append(out_branch_2)
        all_features.append(out_branch_3)

        merged_hidden_states = torch.cat(all_features, dim=1)
        
        output = nn.functional.relu(self.merge_branches_1(merged_hidden_states))
        output = nn.functional.relu(self.merge_branches_2(output))
        output = torch.sigmoid(self.merge_branches_3(output))

        return output
    
    def apply_branch_1(self, x1):
        hidden_state = nn.functional.relu(self.shared_dense_1_1(x1))
        hidden_state = nn.functional.relu(self.shared_dense_1_2(hidden_state))
        hidden_state, _ = self.lstm_1_1(hidden_state)
        hidden_state_relu = nn.functional.relu(hidden_state)
        hidden_state, _ = self.lstm_1_2(hidden_state_relu)
        return nn.functional.relu(hidden_state[:, -1, :]).squeeze(0)

    def apply_branch_2(self, x2):
        hidden_state = nn.functional.relu(self.shared_dense_2_1(x2))
        hidden_state = nn.functional.relu(self.shared_dense_2_2(hidden_state))
        hidden_state, _ = self.lstm_2_1(hidden_state)
        hidden_state_relu = nn.functional.relu(hidden_state)
        hidden_state, _ = self.lstm_2_2(hidden_state_relu)
        hidden_state_relu = nn.functional.relu(hidden_state)
        return nn.functional.relu(hidden_state_relu[:, -1, :]).squeeze(0)

    def apply_branch_3(self, x3):
        hidden_state = nn.functional.relu(self.dense_3_1(x3))
        hidden_state = nn.functional.relu(self.dense_3_2(hidden_state))
        hidden_state = nn.functional.relu(self.dense_3_3(hidden_state))
        return hidden_state

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create sequences for RNN')
    parser.add_argument('ditau_path')
    parser.add_argument('qcd_path')
    parser.add_argument('--model-path','-m',default=None, help='Model path')
    parser.add_argument('--loss_weights', '-l', default=None, type=int, help='Loss weight')
    parser.add_argument('--name', '-n', default=None, help='model name')
    parser.add_argument('--batch_size', '-b', default=500, type=int, help='Batch size')
    parser.add_argument('--patience', '-p', default=10, type=int, help='Patience')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')

    args = parser.parse_args()

    config = {
        "learning_rate": 1e-3,
        "batch_size": args.batch_size,
        "loss_weights": args.loss_weights,
        "name": args.name,
        "patience": args.patience,
    }
    train_run = wandb.init(project="root-gnn", config=config, notes="First train run", tags=["torch", "ltc", "run1", "train"])
    val_run = wandb.init(project="root-gnn", config=config, notes="First run", tags=["torch", "ltc", "run1", "val"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape_1 = (10,6)
    input_shape_2 = (6,4)
    input_shape_3 = (8)

    ditau_data = np.load(args.ditau_path)
    qcd_data = np.load(args.qcd_path)
    all_data = {key: np.concatenate([ditau_data[key], qcd_data[key]]) for key in ditau_data.files}
    ditau_data.close()
    qcd_data.close()
    
    len_data = len(all_data['track_info'])
    track_info = all_data['track_info']
    cluster_info = all_data['cluster_info']
    hlv_info = all_data['hlv_info']
    labels = all_data['labels']
    print(f"Data loaded: {len_data} events")

    train_val_split = 0.12
    indices = np.arange(len_data)
    train_indices, val_indices = train_test_split(indices, test_size=train_val_split, shuffle=True)

    track_train, track_val = track_info[train_indices], track_info[val_indices]
    cluster_train, cluster_val = cluster_info[train_indices], cluster_info[val_indices]
    hlv_train, hlv_val = hlv_info[train_indices], hlv_info[val_indices]
    y_train, y_val= labels[train_indices], labels[val_indices]

    print("Split the data into train and validation sets")


    track_train_tensor = torch.tensor(track_train, dtype=torch.float32)
    track_val_tensor = torch.tensor(track_val, dtype=torch.float32)

    cluster_train_tensor = torch.tensor(cluster_train, dtype=torch.float32)
    cluster_val_tensor = torch.tensor(cluster_val, dtype=torch.float32)

    hlv_train_tensor = torch.tensor(hlv_train, dtype=torch.float32)
    hlv_val_tensor = torch.tensor(hlv_val, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(track_train_tensor, cluster_train_tensor, hlv_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(track_val_tensor, cluster_val_tensor, hlv_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)


    if config['name'] == 'rnn':
        rnn_model = RecurrentEncoder(input_shape_1,input_shape_2,input_shape_3)
    elif config['name'] == 'ltc':
        rnn_model = RecurrentEncoder(input_shape_1, input_shape_2, input_shape_3, lstm_block=LTC)

    if args.model_path:
        model_file = os.path.join(args.model_path, "model.pt")
        if os.path.isfile(model_file):
            print(f"Loading model from {model_file}")
            rnn_model.load_state_dict(torch.load(model_file, map_location=device))

    rnn_model.to(device)

    # TODO: Let an already trained model be loaded and continue training
    if config['loss_weights'] is not None:
        loss = nn.BCELoss(pos_weight=torch.tensor([config['loss_weights']], device=device))
    else:
        loss = nn.BCELoss()
    
    optimizer = optim.Adam(rnn_model.parameters(), lr=config['learning_rate'])

    train_auroc_metric = AUROC(task='binary').to(device)
    val_auroc_metric = AUROC(task='binary').to(device)

    train_run.watch(rnn_model, log_freq=100)
    val_run.watch(rnn_model, log_freq=100)

    # Training Loop
    best_val_loss = float('inf') # for early stopping
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        rnn_model.train()
        running_loss = 0.0
        train_auroc_metric.reset()

        for batch_idx, (track_batch, cluster_batch, hlv_batch, y_batch) in enumerate(train_loader):
            track_batch, cluster_batch, hlv_batch, y_batch = track_batch.to(device), cluster_batch.to(device), hlv_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = rnn_model(track_batch, hlv_batch, cluster_batch)
            loss_value = loss(outputs, y_batch.unsqueeze(1))
            loss_value.backward()
            optimizer.step()
            running_loss += loss_value.item()
            train_auroc_metric.update(outputs, y_batch.int())

            if batch_idx > 0 and batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Train Loss: {running_loss/100:.4f}, Train AUROC: {train_auroc_metric.compute():.4f}")
                running_loss = 0.0

            
        avg_train_loss = running_loss / len(train_loader)
        epoch_train_auroc = train_auroc_metric.compute().item()
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train AUROC: {epoch_train_auroc:.4f}")
        train_run.log({"train_loss": avg_train_loss, "train_auroc": epoch_train_auroc})

        rnn_model.eval()
        val_running_loss = 0.0
        val_auroc_metric.reset()

        with torch.no_grad():
            for batch_idx, (track_batch, cluster_batch, hlv_batch, y_batch) in enumerate(val_loader):
                track_batch, cluster_batch, hlv_batch, y_batch = track_batch.to(device), cluster_batch.to(device), hlv_batch.to(device), y_batch.to(device)

                val_outputs = rnn_model(track_batch, hlv_batch, cluster_batch)
                val_loss_value = loss(val_outputs, y_batch.unsqueeze(1))
                val_auroc_metric.update(val_outputs, y_batch.int())
                val_running_loss += val_loss_value.item()

                if batch_idx > 0 and batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Val Loss: {val_running_loss/100:.4f}, Val AUROC: {val_auroc_metric.compute():.4f}")
                    val_running_loss = 0.0

        avg_val_loss = val_running_loss / len(val_loader)
        epoch_val_auroc = val_auroc_metric.compute().item()
        print(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}, Val AUROC: {epoch_val_auroc:.4f}")
        val_run.log({"val_loss": avg_val_loss, "val_auroc": epoch_val_auroc})

        # model checkpointing (save the best model)
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model to {args.model_path}")
            best_val_loss = avg_val_loss
            if args.model_path:
                if not os.path.isdir(args.model_path):
                    os.makedirs(args.model_path)
                model_file = os.path.join(args.model_path, "model.pt")
                torch.save(rnn_model.state_dict(), model_file)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation loss has not improved from {best_val_loss:.4f} for {epochs_no_improve} epochs")

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement")
            break

    print("Training completed")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if args.model_path is not None:
        print(f"Model saved to {args.model_path}")
        model_file = os.path.join(args.model_path, "model.pt")
        if os.path.exists(model_file):
            train_run.log_artifact(model_file, name='model', type='model')
    else:
        print("Model not saved")

    train_run.finish()
    val_run.finish()

