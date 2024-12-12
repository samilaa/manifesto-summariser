import csv
import os

# Parameters

# Get the absolute path to the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
csv_path = os.path.join(current_dir, 'data', 'raw_answers_and_questions.csv')

input_file = os.path.join(current_dir, 'data', 'raw_answers_and_questions.csv')  # Replace with your input CSV file name
target_party = 'Vasemmistoliitto'  # Replace with the party name you want to filter by
output_file = os.path.join(current_dir, 'outputs', f'filtered_output_{target_party}.csv')  # Replace with your desired output CSV file name

# Open the input file and read it
with open(input_file, newline='', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read the header row
    
    # Identify the index of the "puolue" column
    # This will allow the code to work even if columns are reordered.
    party_index = header.index('puolue')
    
    # Open the output file and prepare to write
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        
        # Write the header to the output
        writer.writerow(header)
        
        # Iterate through rows and write only those that match the target party
        for row in reader:
            if row[party_index] == target_party:
                writer.writerow(row)