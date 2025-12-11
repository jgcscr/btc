import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if X.ndim != 3:
            raise ValueError(f"Expected X to have shape [N, T, F], got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to have shape [N], got {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, F]
        output, (hn, _cn) = self.lstm(x)
        # hn: [num_layers, B, H] -> take last layer's hidden state
        last_hidden = hn[-1]  # [B, H]
        logits = self.fc(last_hidden).squeeze(-1)  # [B]
        return logits


def _load_sequence_dataset(dataset_path: str) -> Dict[str, Any]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)

    X_train_seq = data["X_train_seq"]
    y_train_seq = data["y_train_seq"]
    X_val_seq = data["X_val_seq"]
    y_val_seq = data["y_val_seq"]
    X_test_seq = data["X_test_seq"]
    y_test_seq = data["y_test_seq"]
    feature_names = data["feature_names"].tolist()

    seq_len_arr = data.get("seq_len")
    seq_len = int(seq_len_arr[0]) if seq_len_arr is not None else X_train_seq.shape[1]

    threshold_arr = data.get("threshold")
    threshold = float(threshold_arr[0]) if threshold_arr is not None else 0.0

    return {
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "feature_names": feature_names,
        "seq_len": seq_len,
        "threshold": threshold,
    }


def _evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> Dict[str, Any]:
    model.eval()
    all_probs: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "split": split_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def train_and_evaluate(
    dataset_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 64,
    hidden_size: int = 64,
    lr: float = 1e-3,
    num_layers: int = 1,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = _load_sequence_dataset(dataset_path)
    X_train_seq = dataset["X_train_seq"]
    y_train_seq = dataset["y_train_seq"]
    X_val_seq = dataset["X_val_seq"]
    y_val_seq = dataset["y_val_seq"]
    X_test_seq = dataset["X_test_seq"]
    y_test_seq = dataset["y_test_seq"]
    feature_names = dataset["feature_names"]
    seq_len = dataset["seq_len"]
    threshold_from_ret = dataset["threshold"]

    input_size = X_train_seq.shape[2]

    train_ds = SequenceDataset(X_train_seq, y_train_seq)
    val_ds = SequenceDataset(X_val_seq, y_val_seq)
    test_ds = SequenceDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)

        epoch_loss /= len(train_ds)
        print(f"Epoch {epoch}/{epochs} - train_loss={epoch_loss:.6f}")

    metrics = [
        _evaluate_split(model, train_loader, device, "train"),
        _evaluate_split(model, val_loader, device, "val"),
        _evaluate_split(model, test_loader, device, "test"),
    ]

    model_path = os.path.join(output_dir, "lstm_dir1h_model.pt")
    torch.save(model.state_dict(), model_path)

    metadata = {
        "model_type": "lstm_classifier",
        "feature_names": feature_names,
        "seq_len": seq_len,
        "threshold_label": 0.5,
        "threshold_from_ret": threshold_from_ret,
        "metrics": metrics,
    }

    meta_path = os.path.join(output_dir, "model_metadata_lstm_direction.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved LSTM direction model to:", model_path)
    print("Saved metadata to:", meta_path)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an LSTM-based classifier for 1h BTC direction.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_seq_len24.npz",
        help="Path to the sequence npz file produced by build_sequence_direction_dataset.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/models/lstm_dir1h_v1",
        help="Directory to store the trained LSTM model and metadata",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden-size", type=int, default=64, help="LSTM hidden size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of LSTM layers.")
    args = parser.parse_args()

    train_and_evaluate(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        lr=args.lr,
        num_layers=args.num_layers,
    )


if __name__ == "__main__":
    main()
