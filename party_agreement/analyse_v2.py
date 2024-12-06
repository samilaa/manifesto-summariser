import pandas as pd

# Read the CSV data
df = pd.read_csv('./data/raw_answers_and_questions.csv', encoding='utf-8')

# Get the list of questions
question_cols = df.columns[2:]  # Assuming the first two columns are 'vaalipiiri' and 'puolue'

# Create a DataFrame to store the results
results = []

party_list = df['puolue'].unique()

for party in party_list:
    party_df = df[df['puolue'] == party].copy()
    num_candidates = len(party_df)
    for question in question_cols:
        # Convert the responses to numeric, in case there are any issues
        party_df[question] = pd.to_numeric(party_df[question], errors='coerce')
        mode_value = party_df[question].mode().values[0]
        count_mode = (party_df[question] == mode_value).sum()
        percentage = count_mode / num_candidates * 100
        results.append({
            'puolue': party,
            'question': question,
            'mode_answer': mode_value,
            'percentage_agreement': percentage
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define environmental questions
environmental_questions = [
    'Lihantuotannon tukea tulee vähentää ilmastosyistä.',
    'Suomeen pitää rakentaa ainakin yksi suuri ydinvoimala lisää.',
    'Metsähakkuita pitää rajoittaa ilmastopäästöjä poistavien hiilinielujen kasvattamiseksi.',
    'Valtion pitää ympäristösyistä ohjata ihmisiä kuluttamaan vähemmän.',
    'Suomen pitää suojella kaikki luonnontilaiset metsät, jotta luonnon monimuotoisuus vahvistuisi.',
    'Suomen pitää olla edelläkävijä ilmastonmuutoksen hidastamisessa, vaikka se aiheuttaisi suomalaisille kustannuksia.'
]

# Filter results for environmental questions
env_results = results_df[results_df['question'].isin(environmental_questions)]

# Calculate average agreement for environmental questions per party
env_agreement = env_results.groupby('puolue')['percentage_agreement'].mean().reset_index()
env_agreement.rename(columns={'percentage_agreement': 'average_agreement'}, inplace=True)

print(env_agreement)