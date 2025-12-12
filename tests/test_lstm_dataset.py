import numpy as np
import torch

from src.training.lstm_data import create_dataloader, make_sequences


def test_make_sequences_and_dataloader_shapes() -> None:
    n_rows = 10
    n_features = 4
    seq_len = 3

    X = np.arange(n_rows * n_features, dtype=np.float32).reshape(n_rows, n_features)
    y = (np.arange(n_rows) % 2).astype(np.float32)

    X_seq, y_seq = make_sequences(X, y, seq_len)

    assert X_seq.shape == (n_rows - seq_len + 1, seq_len, n_features)
    assert y_seq.shape == (n_rows - seq_len + 1,)
    # Label should align with the last timestep in each window.
    assert np.array_equal(y_seq, y[seq_len - 1 :])

    loader = create_dataloader(X_seq, y_seq, batch_size=2, shuffle=False)
    batches = list(loader)

    assert len(batches) == int(np.ceil((n_rows - seq_len + 1) / 2))

    first_batch = batches[0]
    X_batch, y_batch = first_batch

    assert isinstance(X_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)
    assert X_batch.shape == (2, seq_len, n_features)
    assert y_batch.shape == (2,)
    # Ensure tensors are float32 and match the expected window contents.
    expected_first_window = torch.from_numpy(X[:seq_len])
    assert torch.allclose(X_batch[0], expected_first_window)
    assert torch.allclose(y_batch, torch.from_numpy(y_seq[:2]))
