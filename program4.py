import spacy
import nltk
import xml.etree.ElementTree as XET
from nltk.corpus import opinion_lexicon
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer


# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load the positive and negative words from the opinion_lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to load the dataset
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

        if aspectTerms:
            dataset[sentence_id] = {
                'text': text,
                'aspect_terms': aspectTerms,
                'predictions': {}
            }
            

    return dataset



# Function to analyze sentiment based on syntactic parsing
def analyze_sentiment_rules(text, aspect_terms):
    doc = nlp(text)

    sentiment_scores = {term: 0 for term in aspect_terms}

    for token in doc:
        if token.text in aspect_terms:
            aspect_term = token.text

            for child in token.children:
                # Rule: If the aspect term's child is a positive word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is positive.
                if (child.text.lower() in positive_words) and (child.dep_ == "amod" or child.dep_ == "advmod"):
                    sentiment_scores[aspect_term] += 1

                

                # Rule: If the aspect term's child is a negative word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is negative.
                elif (child.text.lower() in negative_words ) and (child.dep_ == "amod" or child.dep_ == "advmod"):
                    sentiment_scores[aspect_term] -= 1

            # Analyze sentiment based on the token's parent
            parent = token.head
            # Rule: If the aspect term's parent is a positive word from the opinion_lexicon
            # with dependency type "amod", then the sentiment towards the aspect term is positive.
            if (token.head.text.lower() in positive_words) and (token.dep_ == "amod" or token.dep_ == "advmod"):
                sentiment_scores[aspect_term] += 1

            elif (token.head.text.lower() in negative_words) and (token.dep_ == "amod" or token.dep_ == "advmod"):
                sentiment_scores[aspect_term] -= 1

    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities

    
    
# Function to calculate precision and recall
def calculate_precision_recall(dataset):
    correct_prediction = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    total_prediction = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    total_ground_truth = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }

    for sentence_id, data in dataset.items():
        aspect_terms = data['aspect_terms']
        predictions = data['predictions']

        for aspect_term in aspect_terms:
            if aspect_term['term'] in predictions:
                predicted_polarity = predictions[aspect_term['term']]
                ground_truth_polarity = aspect_term['polarity']

                if predicted_polarity == ground_truth_polarity:
                    correct_prediction[ground_truth_polarity] += 1
                total_prediction[predicted_polarity] += 1
                total_ground_truth.setdefault(ground_truth_polarity, 0)
                total_ground_truth[ground_truth_polarity] += 1

    precision = {
        'positive': (correct_prediction['positive'] / total_prediction['positive']) * 100,
        'negative': (correct_prediction['negative'] / total_prediction['negative']) * 100,
        'neutral': (correct_prediction['neutral'] / total_prediction['neutral']) * 100
    }

    recall = {
        'positive': (correct_prediction['positive'] / total_ground_truth['positive']) * 100,
        'negative': (correct_prediction['negative'] / total_ground_truth['negative']) * 100,
        'neutral': (correct_prediction['neutral'] / total_ground_truth['neutral']) * 100
    }

    return precision, recall



    # Step 1: Load the dataset
dataset= load_data('Restaurants.xml')

# Analyze sentiments and store predictions
for sentence_id, data in dataset.items():
    text = data['text']
    aspect_terms = [term['term'] for term in data['aspect_terms']]
    predictions = analyze_sentiment_rules(text, aspect_terms)
    data['predictions'] = predictions

precision, recall = calculate_precision_recall(dataset)

print("Precision:")
print("Positive:", precision['positive'])
print("Negative:", precision['negative'])
print("Neutral:", precision['neutral'])

print("Recall:")
print("Positive:", recall['positive'])
print("Negative:", recall['negative'])
print("Neutral:", recall['neutral'])