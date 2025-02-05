#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import importlib.resources
import json
from typing import List, Dict

from dlomix.losses import MaskedIonmobLoss
from dlomix.models import Ionmob
from dlomix.data.ion_mobility import get_sqrt_weights_and_biases

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# scikit-learn to split the dataset.
from sklearn.model_selection import train_test_split


class PeptideDataset(Dataset):
    def __init__(self, sequences, mz, charges, targets, targets_2):
        self.sequences = sequences
        self.mz = mz
        self.charges = charges
        self.targets = targets
        self.targets_2 = targets_2

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.sequences[idx],
                self.mz[idx],
                self.charges[idx],
                self.targets[idx],
                self.targets_2[idx])


def tokenize_and_pad(tokens: List[str],
                     tokenizer: Dict[str, int],
                     n: int = 50,
                     pad_position: str = "post") -> np.ndarray:
    """
    Converts a list of tokens to indices using a tokenizer dictionary,
    then pads (or truncates) to length n.
    """
    pad_token = "<PAD>"
    pad_index = tokenizer.get(pad_token, 0)
    token_indices = [tokenizer.get(token, tokenizer.get("<OOV>", 3)) for token in tokens]
    if len(token_indices) < n:
        pad_size = n - len(token_indices)
        padding = [pad_index] * pad_size
        if pad_position == "pre":
            token_indices = padding + token_indices
        else:
            token_indices = token_indices + padding
    else:
        token_indices = token_indices[:n]
    return np.array(token_indices)


def load_tokenizer():
    """Loads the tokenizer from the package resource."""
    with importlib.resources.files("dlomix.data.processing.alphabets").joinpath("unimod-vocab.json").open("r") as f:
        tokenizer_data = json.load(f)
    return tokenizer_data


def create_dataset_from_df(df: pd.DataFrame, tokenizer, max_seq_len: int, num_charges: int = 4) -> PeptideDataset:
    """Converts a DataFrame into a PeptideDataset instance."""
    S = torch.tensor(np.vstack(
        df.sequence_tokenized_unimod.apply(lambda s: tokenize_and_pad(s, tokenizer=tokenizer, n=max_seq_len)).values
    ))
    M = torch.tensor(np.expand_dims(df.mz.astype(np.float32).values, 1))
    # adjust charge to be zero-indexed for one-hot encoding
    C = F.one_hot(torch.tensor(df.charge.values.astype(np.int64) - 1), num_classes=num_charges).float()
    CCS = torch.tensor(np.expand_dims(df.ccs.values, 1).astype(np.float32))
    CCS_PRED = torch.tensor(np.expand_dims(df.ccs_std.values, 1).astype(np.float32))
    return PeptideDataset(S, M, C, CCS, CCS_PRED)


def main():
    parser = argparse.ArgumentParser(description="Train Ionmob model with PyTorch")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to input parquet data file")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate")
    parser.add_argument("--emb_dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--gru1", type=int, default=64,
                        help="First GRU hidden size")
    parser.add_argument("--gru2", type=int, default=32,
                        help="Second GRU hidden size")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="Maximum sequence length for padded sequences")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (in epochs)")
    parser.add_argument("--lr_factor", type=float, default=0.1,
                        help="Learning rate reduction factor")
    parser.add_argument("--lr_patience", type=int, default=2,
                        help="Learning rate scheduler patience (in epochs)")
    parser.add_argument("--min_lr", type=float, default=1e-7,
                        help="Minimum learning rate")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda', 'mps', or 'cpu'. Auto-detect if not specified.")
    parser.add_argument("--print_freq", type=int, default=25,
                        help="Frequency (in steps) to print training loss")
    parser.add_argument("--save_path", type=str, default="best_model.pth",
                        help="Path to save the best trained model")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose mode (print additional information)")
    args = parser.parse_args()

    # verbose: print out configuration settings
    if args.verbose:
        print("Configuration:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    # set device: use provided value or auto-detect
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    print(f"Using device: {device}")

    # load the data from the parquet file.
    data = pd.read_parquet(args.data_path)

    # split data into training, validation, and test sets
    # (train_split + val_split) must be <= 1.0; the remainder is used for testing
    train_val_frac = args.train_split + args.val_split
    if train_val_frac > 1.0:
        raise ValueError("train_split + val_split must be <= 1.0")
    train_val_df, test_df = train_test_split(data, test_size=(1.0 - train_val_frac), random_state=42)
    # compute the relative fraction for validation (from the training+validation portion)
    relative_val_frac = args.val_split / train_val_frac if train_val_frac > 0 else 0.0
    train_df, val_df = train_test_split(train_val_df, test_size=relative_val_frac, random_state=42)

    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

    # load the tokenizer
    unimod_tokenizer = load_tokenizer()

    # create datasets
    train_dataset = create_dataset_from_df(train_df, unimod_tokenizer, args.max_seq_len)
    val_dataset = create_dataset_from_df(val_df, unimod_tokenizer, args.max_seq_len)
    test_dataset = create_dataset_from_df(test_df, unimod_tokenizer, args.max_seq_len)

    # create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # get the weights and biases for the initial sqrt fit using the training data
    weights, biases = get_sqrt_weights_and_biases(
        train_df.mz,
        train_df.charge,
        train_df.ccs,
        fit_charge_state_one=True,
    )

    # create the model and move it to the selected device
    model = Ionmob(
        initial_weights=weights,
        initial_bias=biases,
        emb_dim=args.emb_dim,
        gru_1=args.gru1,
        gru_2=args.gru2,
        num_tokens=len(unimod_tokenizer)
    )
    model.to(device)

    # loss function and optimizer.
    criterion = MaskedIonmobLoss(use_mse=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # learning rate scheduler.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr
    )

    # early stopping tracking.
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        local_step = 0

        for batch in train_loader:
            seq, mz, charge, target_ccs, target_std_ccs = batch
            seq = seq.to(device, dtype=torch.long)
            mz = mz.to(device, dtype=torch.float32)
            charge = charge.to(device, dtype=torch.float32)
            target_ccs = target_ccs.to(device, dtype=torch.float32)
            target_ccs_std = target_std_ccs.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            ccs_predicted, _, ccs_std_predicted = model(seq, mz, charge)
            loss = criterion((ccs_predicted, ccs_std_predicted), (target_ccs, target_ccs_std))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if local_step % args.print_freq == 0:
                avg_loss = running_loss / (local_step + 1)
                print(f"Epoch {epoch}, Step {local_step}, Training Loss: {avg_loss:.4f}")
            local_step += 1

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq, mz, charge, target_ccs, target_std_ccs = batch
                seq = seq.to(device, dtype=torch.long)
                mz = mz.to(device, dtype=torch.float32)
                charge = charge.to(device, dtype=torch.float32)
                target_ccs = target_ccs.to(device, dtype=torch.float32)
                target_ccs_std = target_std_ccs.to(device, dtype=torch.float32)

                ccs_predicted, _, ccs_std_predicted = model(seq, mz, charge)
                loss = criterion((ccs_predicted, ccs_std_predicted), (target_ccs, target_ccs_std))
                val_loss_total += loss.item()
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # learning rate scheduler using the validation loss
        scheduler.step(avg_val_loss)
        if args.verbose:
            current_lr = scheduler.optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}: Current Learning Rate: {current_lr}")

        # early stopping if the validation loss does not improve for a number of epochs, stop
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.")
                break

    # restore the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state.")

    # save the best model to the provided path.
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Best model saved to {args.save_path}")


    model.eval()
    test_loss_total = 0.0
    with torch.no_grad():
        for batch in test_loader:
            seq, mz, charge, target_ccs, target_std_ccs = batch
            seq = seq.to(device, dtype=torch.long)
            mz = mz.to(device, dtype=torch.float32)
            charge = charge.to(device, dtype=torch.float32)
            target_ccs = target_ccs.to(device, dtype=torch.float32)
            target_ccs_std = target_std_ccs.to(device, dtype=torch.float32)

            ccs_predicted, _, ccs_std_predicted = model(seq, mz, charge)
            loss = criterion((ccs_predicted, ccs_std_predicted), (target_ccs, target_ccs_std))
            test_loss_total += loss.item()
    avg_test_loss = test_loss_total / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()