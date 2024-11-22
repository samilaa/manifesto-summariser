import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder 
import sentence_transformers.util
from deep_translator import GoogleTranslator
import torch.serialization  # for fixing numpy serialization error
import numpy as np   
import networkx as nx       # for graph-based clustering
import stanza               # for sentence splitting

# fix for numpy serialization error (occurs, for me at least, when downloading the stanza model)
# i will look into this more later
torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct,
    np.dtype,
    np._globals._NoValue,
    np.core.numeric._frombuffer,
    np.ufunc,
    np.core._exceptions._UFuncNoLoopError
])

# download Finnish stanza model
stanza.download('fi')
nlp = stanza.Pipeline('fi', processors='tokenize', use_gpu=False)

# config
save_path = 'results/funzies.csv'                                     # where to save the translated data
n_test = 100                                                          # how many answers will we process?
top_k = 6                                                             # modify to get k-1 most similar answers for each answer (using cosine similarity)
batch_size = 32
threshold = 0.95                                                      # [0, 1] threshold for whether two sentences are similar                
csv_path = 'data/kuntavaalit2021.csv'
embedding_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')    # a super fast model fine-tuned for semantic search
label_mapping = ['contradiction', 'entailment', 'neutral']            # for interpreting cross-encoder output
cross_encoder = CrossEncoder('cross-encoder/nli-deberta-v3-small')    # TO DO: look into best cross-encoder to use
deepl_api_key = "2076988d-b569-4aa4-8279-194ae522cc4d:fx"       
translator = GoogleTranslator(source='auto', target='en')
transitivity_depth = 3

# for conciceness
def translate(text):
    return translator.translate(text)


"""
Main pipeline for processing answers with sentence-level splitting.
Returns grouped similar answers in Finnish.
"""


# STEP 1: data preparation and sentence splitting
# --------------------------------------------
print("Preparing data...")

# get the explanations from data
df = pd.read_csv(csv_path)
explanations = df['q1.explanation_fi'].dropna().reset_index(drop=True)[:n_test] 

# get the first n_test non-NaN explanations and corresponding answers 
explanation_indices = df['q1.explanation_fi'].dropna().index[:n_test]
answers = df['q1.answer'][explanation_indices]          # save the answers (e.g. likert scale)
explanations_fin = explanations.to_numpy()              # save Finnish versions for later

# track which sentence came from which explanation
sentence_to_original = {}  # sentence index -> corresponding explanations_fin index
all_sentences_fin = []     # Finnish sentences in order of appearance

# split each explanation into sentences
for idx, answer in enumerate(explanations_fin):
    doc = nlp(answer)
    sentences = [sent.text for sent in doc.sentences]
    
    for sentence in sentences:
        sentence_to_original[len(all_sentences_fin)] = idx
        all_sentences_fin.append(sentence)

print("Data preparation completed.")
print(f"Total number of sentences: {len(all_sentences_fin)}.\n")

# STEP 2: Translation and Embedding
# ------------------------------
# translate sentences to English
print("Translating...")

all_sentences_en = []
for i, sentence in enumerate(all_sentences_fin):
    all_sentences_en.append(translate(sentence))
    if i % 50 == 0:
        print(f"Translated {100 * i / len(all_sentences_fin):.2f} %")

print("Translation completed.\n")

# validate translations (i had some issues with the translation API)
# note: this is a very basic check and may not even solve the issue, hasn't been tested yet
for idx, sentence in enumerate(all_sentences_en):
    if sentence is None or not sentence.strip():
        raise ValueError(f"Invalid translated sentence detected at index {idx}: {sentence}")

# create embeddings for sentences
print("Embedding...")
sentence_embeddings = embedding_model.encode(all_sentences_en, convert_to_tensor=True)
print("Embedding completed.\n")

# STEP 3: semantic search 
# -------------------------------------
print("Semantic search...")
hits = sentence_transformers.util.semantic_search( 
    sentence_embeddings, sentence_embeddings, top_k=top_k) # cosine similarity wrapper
print("Semantic search completed.\n")

# STEP 4: cross-encoder verification
# ------------------------------
# create sentence pairs for the cross-encoder
print("Creating sentence pair for cross-encoder...")

query_hit_pairs = []  # [(query_sentence, hit_sentence), ...]
query_hit_pairs_indices = []  # [(query_idx, hit_idx), ...]

for i in range(len(all_sentences_en)):
    for hit in hits[i]:
        if hit['corpus_id'] != i:  # exclude self-matches
            query_hit_pairs.append((all_sentences_en[i], all_sentences_en[hit['corpus_id']]))
            query_hit_pairs_indices.append((i, hit['corpus_id']))

print("Sentence pairs created.\n")

# run sentence pairs through the cross-encoder to see if they are indeed similar
print("Cross-encoder verification...")

# get cross-encoder predictions in batches of 32
predictions = []
for i in range(0, len(query_hit_pairs), batch_size):
    predictions.extend(cross_encoder.predict(query_hit_pairs[i:i+batch_size]))
    if 150 * i % batch_size == 0:
        print(f"Cross-encoder progress: {100 * i / len(query_hit_pairs):.2f} %")

print("Cross-encoder verification completed.\n")

# STEP 5: grouping similar sentences
# ------------------------------
# create groups of similar sentences with references to sentences' parent explanations
grouped_sentences = {}  # (sentence, explanation_idx) -> [(similar_sentence, explanation_idx), ...]

for idx, (prediction, (query_idx, hit_idx)) in enumerate(zip(predictions, query_hit_pairs_indices)):
    if prediction[1] > threshold:  # entailment probability > threshold
        # get the sentences and their parent explanations' indices
        query_sentence = all_sentences_fin[query_idx]
        hit_sentence = all_sentences_fin[hit_idx]
        query_answer_idx = sentence_to_original[query_idx]
        hit_answer_idx = sentence_to_original[hit_idx]
        
        # create a tuple of (sentence, answer_idx) as the key 
        query_key = (query_sentence, query_answer_idx)
        hit_tuple = (hit_sentence, hit_answer_idx)
        
        # add to grouped sentences
        if query_key not in grouped_sentences:
            grouped_sentences[query_key] = []
        grouped_sentences[query_key].append(hit_tuple)

# if the sentence is not in the grouped_sentences, add it as a group of one
for i, sentence in enumerate(all_sentences_fin):
    if (sentence, sentence_to_original[i]) not in grouped_sentences:
        grouped_sentences[(sentence, sentence_to_original[i])] = [(-1, -1)]

print("Grouping similar sentences completed.")

# STEP 5: group similar sentences with limited transitivity 
# (transitivity_depth should and will be experimented with)
# ---------------------------------------------------------

# create a graph
similarity_graph = nx.Graph()

# add nodes (each sentence with its parent answer index is a node)
for i, sentence in enumerate(all_sentences_fin):
    similarity_graph.add_node((sentence, sentence_to_original[i]))

# add an edge between nodes iff. the cross-encoder similarity score is > threshold
for idx, (prediction, (query_idx, hit_idx)) in enumerate(zip(predictions, query_hit_pairs_indices)):
    if prediction[1] > threshold:  # check entailment probability (= similarity score)
        query_node = (all_sentences_fin[query_idx], sentence_to_original[query_idx])
        hit_node = (all_sentences_fin[hit_idx], sentence_to_original[hit_idx])
        similarity_graph.add_edge(query_node, hit_node)

# function to find groups with limited transitivity
def find_bounded_groups(graph, max_path_length=3):
    groups = []
    visited = set()

    # iterate over all nodes in the graph
    for node in graph.nodes:
        if node not in visited:
            # perform a breadth-first search up to transitivity_depth
            bounded_group = set()
            queue = [(node, 0)]  # (current_node, current_depth)
            while queue:
                current_node, depth = queue.pop(0)
                if current_node not in visited and depth <= max_path_length:
                    visited.add(current_node)
                    bounded_group.add(current_node)
                    for neighbor in graph.neighbors(current_node):
                        queue.append((neighbor, depth + 1))
            
            # only keep groups larger than one sentence 
            if len(bounded_group) > 1:
                groups.append(list(bounded_group))
    
    return groups

# find groups with a transitivity limit of transitivity_depth
grouped_sentences = find_bounded_groups(similarity_graph, max_path_length=transitivity_depth)

# results
print(f"Number of groups formed (with transitivity limit): {len(grouped_sentences)}")

# print sentence groups
for i, group in enumerate(grouped_sentences):
    print(f"Group {i+1}:")
    for sentence, original_answer_index in group:
        print(f" -  {sentence}")