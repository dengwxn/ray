#!/bin/bash

# Define the prefix to be removed
actor1="pid=2389736"
actor2="pid=2389738"
logfmtend="\[0m "

# Specify the input and output files
input_file="out_64.log"

cat $input_file | grep $actor1 | sed "s/.*$actor1.*$logfmtend//g" >out_actor1.log
cat $input_file | grep $actor2 | sed "s/.*$actor2.*$logfmtend//g" >out_actor2.log
cat out_actor1.log | grep elapse >out_actor1_elapse.log
cat out_actor2.log | grep elapse >out_actor2_elapse.log
cat header.csv >actor1.csv
cat header.csv >actor2.csv
cat out_actor1_elapse.log | grep -o '[0-9]\+$' | awk '{
    line = (NR % 261 == 1 ? "" : line ",") $1
    if (NR % 261 == 0) {
        print line
        line = ""
    }
} END {
    if (line) print line
}' >>actor1.csv
cat out_actor2_elapse.log | grep -o '[0-9]\+$' | awk '{
    line = (NR % 261 == 1 ? "" : line ",") $1
    if (NR % 261 == 0) {
        print line
        line = ""
    }
} END {
    if (line) print line
}' >>actor2.csv
awk -F',' 'NR == 1 {
    # Print the header row as it is
    print $0
    next
}
{
    # Store the value of the last column
    last_col = $NF
    # Divide each column by the last column
    for (i = 1; i <= NF; i++) {
        $i = ($i / last_col)
    }
    # Print the modified row
    print $0
}' OFS=',' actor1.csv >actor1_percentage.csv
awk -F',' 'NR == 1 {
    # Print the header row as it is
    print $0
    next
}
{
    # Store the value of the last column
    last_col = $NF
    # Divide each column by the last column
    for (i = 1; i <= NF; i++) {
        $i = ($i / last_col)
    }
    # Print the modified row
    print $0
}' OFS=',' actor2.csv >actor2_percentage.csv
awk -F',' '
BEGIN {
    # Initialize group sums for each row
    forward_sum = backward_sum = allreduce_sum = update_sum = 0
}
NR == 1 {
    # Store headers and find the indices for each group
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^forward[0-9]+$/) forward_indices[i]
        if ($i ~ /^backward[0-9]+$/) backward_indices[i]
        if ($i ~ /^allreduce[0-9]+$/) allreduce_indices[i]
        if ($i ~ /^update[0-9]+$/) update_indices[i]
    }
    # Print new headers for summed columns
    print "tensor-to-device,pre-forward,forward_sum,loss,pre-backward,backward_sum,allreduce_sum,update_sum,total"
    next
}
{
    # Reset sums for each row
    forward_sum = backward_sum = allreduce_sum = update_sum = 0

    # Sum values in each group
    for (i in forward_indices) forward_sum += $i
    for (i in backward_indices) backward_sum += $i
    for (i in allreduce_indices) allreduce_sum += $i
    for (i in update_indices) update_sum += $i

    # Output the row with summed columns
    print $1","$2","forward_sum","$66","$67","backward_sum","allreduce_sum","update_sum","$NF
}' "actor2.csv" >"actor2_sum.csv"
awk -F',' '
BEGIN {
    # Initialize group sums for each row
    forward_sum = backward_sum = allreduce_sum = update_sum = 0
}
NR == 1 {
    # Store headers and find the indices for each group
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^forward[0-9]+$/) forward_indices[i]
        if ($i ~ /^backward[0-9]+$/) backward_indices[i]
        if ($i ~ /^allreduce[0-9]+$/) allreduce_indices[i]
        if ($i ~ /^update[0-9]+$/) update_indices[i]
    }
    # Print new headers for summed columns
    print "tensor-to-device,pre-forward,forward_sum,loss,pre-backward,backward_sum,allreduce_sum,update_sum,total"
    next
}
{
    # Reset sums for each row
    forward_sum = backward_sum = allreduce_sum = update_sum = 0

    # Sum values in each group
    for (i in forward_indices) forward_sum += $i
    for (i in backward_indices) backward_sum += $i
    for (i in allreduce_indices) allreduce_sum += $i
    for (i in update_indices) update_sum += $i

    # Output the row with summed columns
    print $1","$2","forward_sum","$66","$67","backward_sum","allreduce_sum","update_sum","$NF
}' "actor1.csv" >"actor1_sum.csv"
awk -F',' 'NR == 1 {
    # Print the header row as it is
    print $0
    next
}
{
    # Store the value of the last column
    last_col = $NF
    # Divide each column by the last column
    for (i = 1; i <= NF; i++) {
        $i = ($i / last_col)
    }
    # Print the modified row
    print $0
}' OFS=',' actor1_sum.csv >actor1_sum_percentage.csv
awk -F',' 'NR == 1 {
    # Print the header row as it is
    print $0
    next
}
{
    # Store the value of the last column
    last_col = $NF
    # Divide each column by the last column
    for (i = 1; i <= NF; i++) {
        $i = ($i / last_col)
    }
    # Print the modified row
    print $0
}' OFS=',' actor2_sum.csv >actor2_sum_percentage.csv

awk -F',' '
NR == 1 {
    # Read the header row to identify columns for allreduce1 to allreduce64
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^allreduce[0-9]+$/) {
            allreduce_columns[i] = 1
            if ($i == "allreduce1") allreduce1_col = i
        }
    }

    # Print the header for the output
    print "Sum of allreduce1 to allreduce64,Allreduce1,Average of allreduce2 to allreduce64"
    next
}
{
    # Initialize variables for the sum and average calculations
    allreduce_sum = 0
    allreduce1 = 0
    allreduce2_to_64_sum = 0
    count = 0

    # Loop through the identified allreduce columns and calculate the required values
    for (i in allreduce_columns) {
        value = $i
        allreduce_sum += value

        if (i == allreduce1_col) {
            allreduce1 = value  # Store the value of allreduce1
        } else {
            allreduce2_to_64_sum += value  # Sum for allreduce2 to allreduce64
            count++
        }
    }

    # Calculate the average for allreduce2 to allreduce64
    allreduce2_to_64_avg = allreduce2_to_64_sum / count

    # Print the results for the current row
    print allreduce_sum "," allreduce1 "," allreduce2_to_64_avg
}' actor1.csv >actor1_allreduce.csv

awk -F',' '
NR == 1 {
    # Read the header row to identify columns for allreduce1 to allreduce64
    for (i = 1; i <= NF; i++) {
        if ($i ~ /^allreduce[0-9]+$/) {
            allreduce_columns[i] = 1
            if ($i == "allreduce1") allreduce1_col = i
        }
    }

    # Print the header for the output
    print "Sum of allreduce1 to allreduce64,Allreduce1,Average of allreduce2 to allreduce64"
    next
}
{
    # Initialize variables for the sum and average calculations
    allreduce_sum = 0
    allreduce1 = 0
    allreduce2_to_64_sum = 0
    count = 0

    # Loop through the identified allreduce columns and calculate the required values
    for (i in allreduce_columns) {
        value = $i
        allreduce_sum += value

        if (i == allreduce1_col) {
            allreduce1 = value  # Store the value of allreduce1
        } else {
            allreduce2_to_64_sum += value  # Sum for allreduce2 to allreduce64
            count++
        }
    }

    # Calculate the average for allreduce2 to allreduce64
    allreduce2_to_64_avg = allreduce2_to_64_sum / count

    # Print the results for the current row
    print allreduce_sum "," allreduce1 "," allreduce2_to_64_avg
}' actor2.csv >actor2_allreduce.csv
