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
    ground_truth = {}

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

        aspectCategory = sentence_elem.find('aspectCategories/aspectCategory')
        if aspectCategory is not None:
            ground_truth_label = aspectCategory.attrib['polarity']
        else:
            ground_truth_label = 'neutral'

        if aspectTerms:
            dataset[sentence_id] = {
                'text': text,
                'aspect_terms': aspectTerms
            }
            ground_truth[sentence_id] = ground_truth_label

    return dataset, ground_truth



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


def determine_sentence_sentiment(sentiment_polarities):
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    for polarity in sentiment_polarities.values():
        if polarity == "positive":
            positive_count += 1
        elif polarity == "negative":
            negative_count += 1
        elif polarity == "neutral":
            neutral_count += 1

    if positive_count > negative_count and positive_count > neutral_count:
        return "positive"
    elif negative_count > positive_count and negative_count > neutral_count:
        return "negative"
    elif neutral_count > positive_count and neutral_count > negative_count:
        return "neutral"
    
    
def analyze_sentiments(dataset):
    predictions = {}

    for sentence_id, data in dataset.items():
        text = data['text']
        aspect_terms = [term['term'] for term in data['aspect_terms']]

        sentiment_polarities = analyze_sentiment_rules(text, aspect_terms)
        sentence_sentiment = determine_sentence_sentiment(sentiment_polarities)

        predictions[sentence_id] = sentence_sentiment

    return predictions


def calculate_precision(predictions, ground_truth, sentiment_category):
    true_positives = 0
    positive_count = 0

    for sentence_id, prediction in predictions.items():
        if prediction == sentiment_category:
            positive_count += 1
            if prediction == ground_truth[sentence_id]:
                true_positives += 1

    precision = (true_positives / positive_count) * 100 if positive_count > 0 else 0.0
    return precision



def calculate_recall(predictions, ground_truth, sentiment_category):
    true_positives = 0
    positive_sentences = 0

    for sentence_id, prediction in predictions.items():
        if ground_truth[sentence_id] == sentiment_category:
            positive_sentences += 1
            if prediction == sentiment_category:
                true_positives += 1

    if positive_sentences == 0:
        recall = 0.0
    else:
        recall = (true_positives / positive_sentences) * 100

    return recall


    # Step 1: Load the dataset
dataset, ground_truth = load_data('Restaurants.xml')

# Analyze sentiments and store predictions
predictions = analyze_sentiments(dataset)
positive_precision = calculate_precision(predictions, ground_truth, "positive")
negative_precision = calculate_precision(predictions, ground_truth, "negative")
neutral_precision = calculate_precision(predictions, ground_truth, "neutral")

print("Positive Precision:", positive_precision)
print("Negative Precision:", negative_precision)
print("Neutral Precision:", neutral_precision)




positive_recall = calculate_recall(predictions, ground_truth, "positive")
negative_recall = calculate_recall(predictions, ground_truth, "negative")
neutral_recall = calculate_recall(predictions, ground_truth, "neutral")

print("Positive Recall:", positive_recall)
print("Negative Recall:", negative_recall)
print("Neutral Recall:", neutral_recall)
# print("Recall:", recall)