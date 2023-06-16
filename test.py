import csv
from decimal import Decimal

# Specify the input and output file paths
input_file = 'results.txt'
output_file = 'output.csv'

# Open the input and output files
with open(input_file, 'r') as file:
    lines = file.readlines()

with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['x', 'y', 'z'])  # Write the header row

    for line in lines:
        if len(line) <= 2:
            continue
        # Remove the square brackets and split the line by the closing bracket and any remaining whitespace
        data = line.split(',')
        x = float(data[0])
        y = float(data[1])
        z = float(data[2])

        # Write the values into the CSV file
        writer.writerow([x, y, z])
