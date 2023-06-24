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

        if aspectTerms:
            dataset[sentence_id] = {
                'text': text,
                'aspect_terms': aspectTerms
            }
            

    return dataset



# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load the positive words from the opinion_lexicon
positive_words = set(opinion_lexicon.positive())

# Load the Negative words from the opinion_lexicon
negative_words = set(opinion_lexicon.negative())

positive_words_adverbs = ['beautifully', 'happily', 'wonderfully']
negative_words_adverbs = ['sadly', 'badly', 'poorly']


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
                if (child.text.lower() in positive_words or child.text.lower() in positive_words_adverbs) and (child.dep_ == "amod" or child.dep_ == "advmod"):
                    sentiment_scores[aspect_term] += 1

                

                # Rule: If the aspect term's child is a negative word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is negative.
                elif (child.text.lower() in negative_words or child.text.lower() in negative_words_adverbs) and (child.dep_ == "amod" or child.dep_ == "advmod"):
                    sentiment_scores[aspect_term] -= 1

            # Analyze sentiment based on the token's parent
            parent = token.head
            # Rule: If the aspect term's parent is a positive word from the opinion_lexicon
            # with dependency type "amod", then the sentiment towards the aspect term is positive.
            if (token.head.text.lower() in positive_words or token.head.text.lower() in positive_words_adverbs) and (token.dep_ == "amod" or token.dep_ == "advmod"):
                sentiment_scores[aspect_term] += 1

            elif (token.head.text.lower() in negative_words or token.head.text.lower() in negative_words_adverbs) and (token.dep_ == "amod" or token.dep_ == "advmod"):
                sentiment_scores[aspect_term] -= 1

    # Determine sentiment polarities based on the sentiment scores
    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities




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

# Usage example
# file_path = 'Restaurants.xml'
file_path = 'rest.xml'
dataset = load_data(file_path)

test1(dataset)

# print(dataset)