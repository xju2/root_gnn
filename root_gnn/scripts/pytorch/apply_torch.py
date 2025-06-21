#!/usr/bin/env python

from train_torch import RecurrentEncoder
print("Imported RecurrentEncoder")

import argparse
import numpy as np
import os


import torch
from torch import tensor
from torch import load
from torch.nn import BCELoss
from torch.utils.data import DataLoader, TensorDataset

from torch.nn import LSTM
from ncps.torch import LTC
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchmetrics import AUROC, Accuracy


# npz in, npz out, model in

def evaluate(model, dataloader, device, loss_func, incl_clusters=True, run=None):
    model.eval()

    total_loss = 0
    all_predictions = []
    all_labels = []

    auroc = AUROC(task="binary").to(device)
    accuracy = Accuracy(task="binary").to(device)

    with torch.no_grad(): # don't need to do gradient calculations for inference
        for batch_data in dataloader:
            track_info, cluster_info, hlv_info, labels_info = batch_data

            track_info = track_info.to(device)
            cluster_info = cluster_info.to(device)
            hlv_info = hlv_info.to(device)
            labels_info = labels_info.to(device)

            outputs = model(x1 = track_info, x2 = cluster_info if incl_clusters else None, x3 = hlv_info)

            loss = loss_func(outputs, labels_info)
            total_loss += loss.item()

            auroc.update(outputs, labels_info.int())
            accuracy.update(outputs, labels_info.int())

            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels_info.cpu().numpy())

            if run is not None:
                run.log({"test_loss": loss.item(), "test_auroc": auroc.compute().item(), "test_accuracy": accuracy.compute().item()})

    avg_loss = total_loss / len(dataloader)
    final_metrics = {'loss': avg_loss, 'auroc': auroc.compute().item(), 'accuracy': accuracy.compute().item()}

    predictions = np.concatenate(all_predictions, axis=0)
    return predictions.squeeze(), final_metrics
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Predict with RNN')
    parser.add_argument('model_path', default=None)
    parser.add_argument('--input','-i',default=None)
    parser.add_argument('--output','-o',default=None)
    parser.add_argument('--loss_weights','-l', default=1, type=float)
    parser.add_argument('--name', '-n', default="lstm")

    args = parser.parse_args()
    data = np.load(args.input)


    config = {
        "name": args.name,
        "model_path": args.model_path,
        "input": args.input,
        "output": args.output,
    }
    run = wandb.init(project="root-gnn", config=config, notes="First run", tags=["torch", "ltc", "run1", "apply"])

    input_shape_1 = (10, 6)
    input_shape_2 = (6, 4)
    input_shape_3 =  8

    if args.name == "ltc":
        lstm_block = LTC
        ltc_block = True
    else:
        lstm_block = LSTM
        ltc_block = False

    model = RecurrentEncoder(input_shape_1, input_shape_2, input_shape_3, rnn_block=lstm_block, ltc_block=ltc_block)
    model_file = os.path.join(args.model_path, "model.pt")
    model.load_state_dict(torch.load(model_file, map_location=device))

    model.to(device)

    batch_size = 500
    if len(data['track_info']) < batch_size:
        batch_size = len(data['track_info'])

    
    n_events = len(data['track_info'])
    len_data = (n_events // batch_size) * batch_size

    track_tensor = torch.tensor(data['track_info'][:len_data], dtype=torch.float32, device=device)
    cluster_tensor = torch.tensor(data['cluster_info'][:len_data], dtype=torch.float32, device=device)
    hlv_tensor = torch.tensor(data['hlv_info'][:len_data], dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(data['labels'][:len_data], dtype=torch.float32, device=device)

    dataset = TensorDataset(track_tensor, cluster_tensor, hlv_tensor, labels_tensor)
    inference_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if args.loss_weights is not None:
        loss_func = torch.nn.BCELoss(weight=torch.tensor([args.loss_weights], device=device))
    else:
        loss_func = torch.nn.BCELoss()

    predictions, metrics = evaluate(model, inference_loader, device, loss_func, run=run)


    print("\n--- Evaluation Results ---")
    print(f"  Test Loss: {metrics.get('loss', 'N/A'):.4f}")
    print(f"  Test Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
    print(f"  Test AUROC:    {metrics.get('auroc', 'N/A'):.4f}")
    print("--------------------------\n") 
    np.savez(args.output,track=track_tensor.cpu().numpy(), cluster=cluster_tensor.cpu().numpy(), hlv=hlv_tensor.cpu().numpy(), predictions=predictions,truth_info=labels_tensor.cpu().numpy())
    run.finish()
