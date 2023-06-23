# Student ID: a1802961
# Name      : Vinay Kumar


# Library imports.....
import nltk
# nltk.download('opinion_lexicon')
import xml.etree.ElementTree as XET
import spacy
from nltk.corpus import opinion_lexicon



#Lib imports ends.....

# -------------Load the data--------------

def load_data(path):
    tree = XET.parse(path)
    root = tree.getroot()
    dataset = {}

    for sentence_elem in root.iter('sentence'):
        sentence_id = sentence_elem.attrib['id']
        text = sentence_elem.find('text').text

        aspectTerms = []
        aspect_terms_elem = sentence_elem.find('aspectTerms')

        if aspect_terms_elem is not None:
            for aspect_term_elem in aspect_terms_elem.iter('aspectTerm'):
                aspect_term = aspect_term_elem.attrib['term']
                polarity = aspect_term_elem.attrib['polarity']
                aspectTerms.append({
                    'term': aspect_term,
                    'polarity': polarity
                })

        dataset[sentence_id] = {
            'text': text,
            'aspect_terms': aspectTerms
        }
            

    return dataset

#-------------Syntactic Parsing-----------
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load the positive words from the opinion_lexicon
positive_words = set(opinion_lexicon.positive())

# Load the Negative words from the opinion_lexicon
negative_words = set(opinion_lexicon.negative())

# Function to analyze sentiment based on syntactic parsing
def analyze_sentiment(text, aspect_terms):
    doc = nlp(text)

    # Initialize sentiment scores for each aspect term
    sentiment_scores = {term: 0 for term in aspect_terms}

    for token in doc:
        # Check if the token is an aspect term
        if token.text in aspect_terms:
            aspect_term = token.text

            # Analyze sentiment based on the token's children and dependencies
            for child in token.children:
                # Rule: If the aspect term's child is a positive word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is positive.
                if child.text in positive_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] += 1

                # Rule: If the aspect term's child is a negative word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is negative.
                elif child.text in negative_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] -= 1

                # Rule: If the aspect term's child has dependency type "amod" but is not a positive or negative word,
                # then the sentiment towards the aspect term is neutral.
                elif child.dep_ == "amod":
                    sentiment_scores[aspect_term] += 0

    # Determine sentiment polarities based on the sentiment scores
    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities



#-------------Precision calculator-----------

# Function to store predicted polarities and calculate precision and recall
def evaluate_polarities(dataset):
    # Initialize counters for precision and recall calculation
    true_positive = 0
    detected_positive = 0
    total_positive = 0

    for sentence_id, data in dataset.items():
        text = data['text']
        aspect_terms = [aspect['term'] for aspect in data['aspect_terms']]
        sentiment_polarities = analyze_sentiment(text, aspect_terms)
        polarities = {aspect['term']: aspect['polarity'] for aspect in data['aspect_terms']}

        for aspect_term, predicted_polarity in sentiment_polarities.items():
            if predicted_polarity == 'positive':
                detected_positive += 1
                if polarities[aspect_term] == 'positive':
                    true_positive += 1
            if polarities[aspect_term] == 'positive':
                total_positive += 1

        # Store the predicted polarities in the dataset
        dataset[sentence_id]['predicted_polarities'] = sentiment_polarities

    # Calculate precision and recall
    precision = true_positive / detected_positive if detected_positive > 0 else 0
    recall = true_positive / total_positive if total_positive > 0 else 0

    return dataset, precision, recall


#-----tesing Anzalyze -----

def test1(dataset):
    # Initialize counters for precision and recall calculation
    

    for sentence_id, data in dataset.items():
        text = data['text']
        aspect_terms = [aspect['term'] for aspect in data['aspect_terms']]
        sentiment_polarities = analyze_sentiment(text, aspect_terms)
        polarities = {aspect['term']: aspect['polarity'] for aspect in data['aspect_terms']}

        print(sentiment_polarities)

        # Store the predicted polarities in the dataset
        dataset[sentence_id]['predicted_polarities'] = sentiment_polarities

   

    return dataset


# -------------Main--------------

# Usage example
file_path = 'rest.xml'
dataset = load_data(file_path)


# Evaluate polarities and calculate precision and recall
test1(dataset)

print(negative_words)

# Print precision and recall
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# # Print the loaded data
# print(data["2777"])







