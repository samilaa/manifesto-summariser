import pandas as pd

# Load the dataset
file_path = './data/raw_answers_and_questions.csv'
df = pd.read_csv(file_path)

# Extract relevant columns: party and answers
party_column = 'puolue'
answer_columns = df.columns[2:]  # Assuming the answers start from the third column

# Function to calculate agreement percentage within each party
def calculate_agreement(df, party_column, answer_columns):
    disagreement_summary = {}
    
    # Group by party
    party_groups = df.groupby(party_column)
    
    for party, group in party_groups:
        total_candidates = len(group)
        disagreement_count = {}
        
        for question in answer_columns:
            # Calculate mode (most common answer)
            mode_answer = group[question].mode()[0]
            
            # Calculate number of candidates who disagree with the mode answer
            num_disagree = (group[question] != mode_answer).sum()
            disagreement_percentage = (num_disagree / total_candidates) * 100
            
            disagreement_count[question] = disagreement_percentage
        
        disagreement_summary[party] = disagreement_count
    
    return disagreement_summary

# Calculate disagreement summary
disagreement_summary = calculate_agreement(df, party_column, answer_columns)

# Display disagreement summary
for party, questions in disagreement_summary.items():
    print(f"Party: {party}")
    for question, disagreement in questions.items():
        print(f"  Question '{question}': {disagreement:.2f}% disagreement")
    print()