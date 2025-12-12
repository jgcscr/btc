import argparse

from src.training.lstm_data import save_sequence_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sequence (temporal) direction dataset from flat direction splits.",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_splits.npz",
        help="Path to the flat direction npz file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="artifacts/datasets/btc_features_1h_direction_seq_len24.npz",
        help="Path to save the sequence npz file.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=24,
        help="Sequence length (number of timesteps) per example.",
    )
    args = parser.parse_args()

    save_sequence_dataset(args.input_path, args.output_path, args.seq_len)


if __name__ == "__main__":
    main()
