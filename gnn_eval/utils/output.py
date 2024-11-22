import torch
import numpy as np
import pandas as pd
import time


def generate_dataframe(model, dataloader, loss_fn, scaler, device, num_classes=2, classification=True):
    """
    Generate a DataFrame from model predictions.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): DataLoader providing input data.
        loss_fn (callable): Loss function for computing losses.
        scaler (object or None): Scaler for inverse transforming outputs (if regression).
        device (str): Device for computation ('cuda' or 'cpu').
        classification (bool): Whether the task is classification or regression.

    Returns:
        pd.DataFrame: DataFrame containing predictions, losses, and other relevant info.
    """
    
    model.eval()  # Ensure model is in evaluation mode
    start_time = time.time()
    with torch.no_grad():
        # Initialize an empty dataframe with columns
        df = pd.DataFrame(columns=['id', 'name', 'loss', 'real', 'pred', 'output'])
        for d in dataloader:
            d.to(device)
            outputs = model(d).cpu()
            y_trues = d.y.cpu()  # Assuming your dataloader provides a 'target' key

            losses = loss_fn(outputs, y_trues).cpu().numpy()  # Compute loss for the batch
            # print('outputs, y_trues, losses: ', outputs.shape, y_trues.shape, losses.shape)
            # Scaling and classification adjustments for the whole batch
            reals = y_trues.numpy()
            outs = outputs.numpy()
            preds = outputs.numpy()
            if scaler is not None:
                reals = scaler.inverse_transform(reals)
                preds = scaler.inverse_transform(preds)
            if classification:
                # preds = (preds >= 0.5)
                if num_classes == 2:
                    preds = (preds >= 0.5)
                else:
                    preds = np.argmax(preds, axis=-1, keepdims=True)
                    losses = loss_fn(outputs, y_trues.flatten().long()).cpu().numpy() 

            # Assuming 'id' and 'symbol' are lists of ids and symbols for each instance in the batch
            batch_ids = d['id']
            batch_names = d['symbol']
            # Create a temporary dataframe for the current batch
            for i in range(len(batch_ids)):
                temp_df = pd.DataFrame({
                    'id': [batch_ids[i]],
                    'name': [batch_names[i]],
                    'loss': [losses[i]],
                    'real': [reals[i]],
                    'pred': [preds[i]],
                    'output': [outs[i]]
                })
                df = pd.concat([df, temp_df], ignore_index=True)
    return df