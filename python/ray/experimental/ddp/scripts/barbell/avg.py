import os
import pandas as pd
import argparse
import sys


def calculate_average_later_runs(input_folder, output_file_rank0, output_file_rank1):
    # List all files in the input folder
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".csv")])

    # Separate files for rank 0 and rank 1
    rank0_files = [os.path.join(input_folder, f) for f in files if "rank0" in f]
    rank1_files = [os.path.join(input_folder, f) for f in files if "rank1" in f]

    # Ensure there are 50 pairs of files
    if len(rank0_files) != 50 or len(rank1_files) != 50:
        print("Error: There should be exactly 50 files for each rank.")
        sys.exit(1)

    # Take the later 40 runs (ignore first 10)
    rank0_files = rank0_files[10:]
    rank1_files = rank1_files[10:]

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
        "--output-path", type=str, help="Path to the output folder for the averaged CSVs."
    )

    args = parser.parse_args()

    output_file_rank0 = os.path.join(args.output_path, "avg_latency_rank0.csv")
    output_file_rank1 = os.path.join(args.output_path, "avg_latency_rank1.csv")

    calculate_average_later_runs(
        args.input_folder, output_file_rank0, output_file_rank1
    )
    print(f"Averaged files saved as {output_file_rank0} and {output_file_rank1}.")
