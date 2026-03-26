"""
Generate train/val/test CSV files from metadataV0.csv for SlowFast training.

This script creates separate CSV files for each split (train, val, test)
based on the SplitA or SplitB column in metadataV0.csv.
"""

import pandas as pd
from pathlib import Path


def generate_split_csvs(
    input_path: str,
    output_dir: str,
    split_column: str = "SplitA"
):
    """
    Generate train/val/test CSV files from metadata.

    Args:
        input_path: Path to metadataV0.csv
        output_dir: Directory to save split CSV files
        split_column: Column name for split assignment (SplitA or SplitB)
    """
    # Load metadata
    print(f"Loading metadata from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Total samples: {len(df)}")
    print(f"Split distribution ({split_column}):")
    print(df[split_column].value_counts())

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate train.csv
    train_df = df[df[split_column] == "train"].reset_index(drop=True)
    train_output = output_dir / "train.csv"
    train_df.to_csv(train_output, index=False)
    print(f"\nSaved {len(train_df)} training samples to: {train_output}")

    # Generate val.csv (same as train for validation during training)
    val_df = df[df[split_column] == "train"].reset_index(drop=True)
    val_output = output_dir / "val.csv"
    val_df.to_csv(val_output, index=False)
    print(f"Saved {len(val_df)} validation samples to: {val_output}")

    # Generate test.csv
    test_df = df[df[split_column] == "test"].reset_index(drop=True)
    test_output = output_dir / "test.csv"
    test_df.to_csv(test_output, index=False)
    print(f"Saved {len(test_df)} test samples to: {test_output}")

    # Print statistics
    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{split_name.upper()} Split ({len(split_df)} samples):")
        print("-" * 40)

        if len(split_df) > 0:
            print(f"Branch distribution:")
            print(split_df['Branch'].value_counts().to_string())
            print(f"\nPlaque statistics:")
            print(f"  Mean: {split_df['Plaque'].mean():.2f}")
            print(f"  Std:  {split_df['Plaque'].std():.2f}")
            print(f"  Min:  {split_df['Plaque'].min():.2f}")
            print(f"  Max:  {split_df['Plaque'].max():.2f}")

    print("\n" + "=" * 60)
    print("CSV files generated successfully!")
    print("=" * 60)
    print(f"\nTo use SplitB instead, run with split_column='SplitB'")


if __name__ == "__main__":
    # File paths
    input_file = r"E:\pycharm23\Projs\DcmDataset\Central\metadataV0.csv"
    output_directory = r"E:\pycharm23\Projs\DcmDataset\Central"

    # Generate splits for SplitA
    print("=" * 60)
    print("Generating splits using SplitA column")
    print("=" * 60)
    generate_split_csvs(input_file, output_directory, split_column="SplitA")

    # Generate splits for SplitB (optional)
    print("\n" + "=" * 60)
    print("Generating splits using SplitB column")
    print("=" * 60)
    generate_split_csvs(input_file, output_directory, split_column="SplitB")
