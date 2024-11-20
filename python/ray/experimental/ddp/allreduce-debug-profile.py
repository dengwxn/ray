import pstats

# Load the profiling data
stats = pstats.Stats("allreduce.prof")

# Sort by cumulative time and print the top 10 results
stats.sort_stats("cumulative").print_stats()
