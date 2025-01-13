import os
import pandas as pd
import argparse
import sys


def calculate_average_later_runs(
    input_folder: str,
    output_file_rank0: str,
    output_file_rank1: str,
    num_trials: int,
    skip_trials: int,
) -> None:
    # List all files in the input folder
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])

    # Separate files for rank 0 and rank 1
    rank0_files = [os.path.join(input_folder, f) for f in files if "rank0" in f]
    rank1_files = [os.path.join(input_folder, f) for f in files if "rank1" in f]

    # Ensure there are `num_trials` pairs of files
    if len(rank0_files) != num_trials or len(rank1_files) != num_trials:
        print(f"Error: There should be exactly {num_trials} files for each rank.")
        sys.exit(1)

    # Take the later runs (ignore first `skip_trials`)
    rank0_files = rank0_files[skip_trials:]
    rank1_files = rank1_files[skip_trials:]

    # Read and average the data for rank 0
    df_rank0_list = [pd.read_csv(f) for f in rank0_files]
    avg_rank0 = pd.concat(df_rank0_list).groupby("name", as_index=False).mean()

    # Read and average the data for rank 1
    df_rank1_list = [pd.read_csv(f) for f in rank1_files]
    avg_rank1 = pd.concat(df_rank1_list).groupby("name", as_index=False).mean()

    # Save the averaged results to output files
    avg_rank0.to_csv(output_file_rank0, index=False)
    avg_rank1.to_csv(output_file_rank1, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate averages for the later 40 runs of CSV data."
    )
    parser.add_argument(
        "--input-folder", type=str, help="Path to the folder containing the CSV files."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output folder for the averaged CSVs.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        help="Number of trials for each rank.",
    )
    parser.add_argument(
        "--skip-trials",
        type=int,
        help="Number of trials to skip.",
    )

    args = parser.parse_args()

    output_file_rank0 = os.path.join(args.output_path, "avg_latency_rank0.csv")
    output_file_rank1 = os.path.join(args.output_path, "avg_latency_rank1.csv")

    calculate_average_later_runs(
        args.input_folder,
        output_file_rank0,
        output_file_rank1,
        args.num_trials,
        args.skip_trials,
    )
    print(f"Averaged files saved as {output_file_rank0} and {output_file_rank1}.")
