import pandas as pd
import os
import sys
# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from chatbot_api.providers.openai import OpenAIProvider
from chatbot_api.base import EmbeddingResponse, Message, Role
from dotenv import load_dotenv
from pathlib import Path
import asyncio

# Load environment variables
env_path = Path(__file__).resolve().parents[1] / ".env" # move up one directory to find the .env file
success = load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("OPENAI_API_KEY")

provider = OpenAIProvider(api_key=key)

# Get the absolute path to the directory containing this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the CSV file
csv_path = os.path.join(current_dir, 'data', 'raw_answers_and_questions.csv')

# Load the CSV file
df = pd.read_csv(csv_path, encoding='utf-8')

# Get the list of questions
question_cols = df.columns[2:]  # Assuming the first two columns are 'vaalipiiri' and 'puolue'

# Define question categories
question_categories = {
    'Environment': [
        'Lihantuotannon tukea tulee vähentää ilmastosyistä.',
        'Suomeen pitää rakentaa ainakin yksi suuri ydinvoimala lisää.',
        'Metsähakkuita pitää rajoittaa ilmastopäästöjä poistavien hiilinielujen kasvattamiseksi.',
        'Valtion pitää ympäristösyistä ohjata ihmisiä kuluttamaan vähemmän.',
        'Suomen pitää suojella kaikki luonnontilaiset metsät, jotta luonnon monimuotoisuus vahvistuisi.',
        'Suomen pitää olla edelläkävijä ilmastonmuutoksen hidastamisessa, vaikka se aiheuttaisi suomalaisille kustannuksia.',
        'Valtion pitää lopettaa maakuntien lentoliikenteen tukeminen.',
        'Bensan ja dieselin verotusta on alennettava.'
    ],
    'Economy': [
        'Suomessa on liian helppo elää yhteiskunnan tukien varassa.',
        'Pääomatulojen verotusta on kiristettävä.',
        'Palkoista pitäisi sopia ensisijaisesti työpaikoilla.',
        'Ansiosidonnaisen työttömyysturvan kestoa pitää lyhentää.',
        'Kun valtion menoja ja tuloja tasapainotetaan, se on tehtävä mieluummin menoja vähentämällä kuin veroja korottamalla.',
        'Valtion on mieluummin otettava lisää velkaa kuin vähennettävä palveluita.',
        'Kehitysyhteistyöhön käytettäviä varoja on leikattava.',
        'Kiina-riippuvuudesta pitää pyrkiä eroon, vaikka se heikentäisi yritysten kilpailukykyä.',
        'Korkeakouluissa on otettava käyttöön lukukausimaksut myös suomalaisille opiskelijoille.',
        'Lapsilisiä pitäisi maksaa vähemmän perheille, joilla on suuret tulot.',
        'Valtion pitää maksaa osa opintolainasta, jos korkeakoulusta vastavalmistunut muuttaa töihin kasvukeskusten ulkopuolelle.'
    ],
    'Education': [
        'Liikuntaa on lisättävä peruskoulussa, vaikka muiden oppiaineiden kustannuksella.',
        'Koulutuksen rahoitusta on lisättävä ensi vaalikaudella.',
        'Pääsykokeiden painoarvoa tulisi lisätä korkeakoulujen opiskelijavalinnoissa.',
        'Ruotsin kielen opiskelu on muutettava vapaaehtoiseksi kaikilla koulutusasteilla.',
        'Ruotsin kielen asema Suomessa on säilytettävä vähintään ennallaan.',
        'Korkeakouluissa on otettava käyttöön lukukausimaksut myös suomalaisille opiskelijoille.'
    ],
    'Social Issues': [
        'Suomen pitää ottaa käyttöön kolmas virallinen sukupuoli.',
        'Yhteiskunnan johtavissa asemissa olevat eivät ymmärrä kansan ongelmia.',
        'Asevelvollisuuden pitää koskea kaikkia sukupuolesta riippumatta.',
        'Kannabiksen käyttö pitää laillistaa.',
        'Suomeen ei pidä avata huumeiden käyttöhuoneita.',
        'Viinin myynti pitää sallia jatkossakin vain Alkossa.',
        'Turkistarhaus pitää sallia myös tulevaisuudessa.',
        'Kansanäänestyksiä on lisättävä, jotta kansalaisten suora osallistuminen päätöksentekoon kasvaa.'
    ],
    'Welfare and Social Services': [
        'On hyväksyttävää, että julkisia palveluja on vähemmän syrjäseuduilla.',
        'Sosiaali- ja terveyspalvelut on tuotettava ensisijaisesti julkisina palveluina.',
        'Eläkeläisköyhyyttä pitää vähentää korottamalla takuueläkettä.',
        'Vanhuspalvelujen hoitajamitoitusta pitää keventää, jotta vanhuksille riittäisi hoivapaikkoja.',
        'Terapian tulee olla maksutonta alle 30-vuotiaille.'
    ],
    'Foreign Policy and Security': [
        'Suomeen tulisi sijoittaa pysyvä Naton tukikohta.',
        'Päätösvaltaa pitäisi siirtää EU:lta jäsenvaltioille.',
        'Ukraina pitää hyväksyä pikimmiten EU:n jäseneksi, vaikka se ei täyttäisi vielä jäsenyyskriteereitä.',
        'Suomen on liityttävä Natoon.',
        'Ahvenanmaan demilitarisoinnista pitäisi luopua.',
        'Kiina-riippuvuudesta pitää pyrkiä eroon, vaikka se heikentäisi yritysten kilpailukykyä.'
    ],
    'Immigration': [
        'Suomen pitää vastaanottaa nykyistä vähemmän kiintiöpakolaisia pakolaisleireiltä.',
        'Työperäistä maahanmuuttoa tarvitaan suomalaisen hyvinvointiyhteiskunnan ylläpitämiseksi.',
        'Yrityksillä pitää olla vapaus palkata työntekijöitä EU:n ulkopuolelta ilman saatavuusharkintaa.',
        'Suomen pitää pystyä karkottamaan nykyistä helpommin rikoksia tehneitä maahanmuuttajia.'
    ],
    'Health Care': [
        'Terapian tulee olla maksutonta alle 30-vuotiaille.',
        'Sosiaali- ja terveyspalvelut on tuotettava ensisijaisesti julkisina palveluina.',
        'Vanhuspalvelujen hoitajamitoitusta pitää keventää, jotta vanhuksille riittäisi hoivapaikkoja.',
        'Suomeen ei pidä avata huumeiden käyttöhuoneita.'
    ],
    'Language Policy': [
        'Ruotsin kielen opiskelu on muutettava vapaaehtoiseksi kaikilla koulutusasteilla.',
        'Ruotsin kielen asema Suomessa on säilytettävä vähintään ennallaan.'
    ],
    'Energy': [
        'Suomeen pitää rakentaa ainakin yksi suuri ydinvoimala lisää.'
    ],
    'Governance': [
        'Kansanäänestyksiä on lisättävä, jotta kansalaisten suora osallistuminen päätöksentekoon kasvaa.'
    ],
    'Animal Rights': [
        'Turkistarhaus pitää sallia myös tulevaisuudessa.'
    ]
}

# Function to map responses to categories
def map_response_to_category(response):
    if response in [4, 5]:
        return 'Agree'
    elif response == 3:
        return 'Neutral'
    elif response in [1, 2]:
        return 'Disagree'
    else:
        return 'Unknown'  # For any unexpected values

async def main():
    # Create a list to store the results
    results = []

    party_list = df['puolue'].unique()

    for party in party_list:
        party_df = df[df['puolue'] == party].copy()
        num_candidates = len(party_df)
        for question in question_cols:
            # Convert the responses to numeric
            party_df[question] = pd.to_numeric(party_df[question], errors='coerce')
            # Map responses to categories
            party_df['response_category'] = party_df[question].apply(map_response_to_category)
            # Get the mode category
            mode_category = party_df['response_category'].mode().values[0]
            count_mode = (party_df['response_category'] == mode_category).sum()
            percentage = count_mode / num_candidates * 100
            results.append({
                'puolue': party,
                'question': question,
                'mode_category': mode_category,
                'percentage_agreement': percentage
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Now, you can analyze agreement for all categories
    # For example, to get agreement for 'Environment' questions:

    for category, questions in question_categories.items():
        category_results = results_df[results_df['question'].isin(questions)]
        # Calculate average agreement per party for this category
        category_agreement = category_results.groupby('puolue')['percentage_agreement'].mean().reset_index()
        category_agreement.rename(columns={'percentage_agreement': 'average_agreement'}, inplace=True)
        category_agreement['category'] = category
        # print(f"\nAverage Agreement in {category} Category:")
        # print(category_agreement)

    # Optionally, combine all category agreements into one DataFrame
    all_category_agreements = []

    for category, questions in question_categories.items():
        category_results = results_df[results_df['question'].isin(questions)]
        category_agreement = category_results.groupby('puolue')['percentage_agreement'].mean().reset_index()
        category_agreement.rename(columns={'percentage_agreement': 'average_agreement'}, inplace=True)
        category_agreement['category'] = category
        all_category_agreements.append(category_agreement)

    # Define the output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the CSV file
    output_path = os.path.join(output_dir, 'overall_agreement_across_categories.csv')

    # Concatenate all category agreements
    all_agreements_df = pd.concat(all_category_agreements, ignore_index=True)
    all_agreements_df.to_csv(output_path, index=False, encoding='utf-8')

    prompt = 'Please summarise these results: ' + all_agreements_df.to_string()

    print('Estimated cost:', provider.estimated_cost(await provider.count_tokens(prompt), 500.0))
    # cost of sending the manifesto: 0.046865500000000004 (of ? currency)

    msg = [Message(role=Role("user"), content=prompt)]

    response = await provider.generate(
        msg,
        temperature=0.7,
    )

    # Display the combined agreements
    # print("\nOverall Agreement Across Categories:")
    print(response.content)
    print(all_agreements_df)

if __name__ == "__main__":
    asyncio.run(main())