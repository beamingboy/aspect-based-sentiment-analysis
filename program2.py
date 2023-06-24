# Student ID: a1802961
# Name      : Vinay Kumar


# Library imports.....
import spacy
import nltk
import xml.etree.ElementTree as XET
from nltk.corpus import opinion_lexicon


# ----------Initializes the resources----------

# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load the positive and negative words from the opinion_lexicon
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# List of intensifiers and negation words
intensifiers = ["very", "extremely", "highly", "remarkably", "exceptionally", "incredibly", "immensely", "exceedingly", "tremendously", "intensely", "extraordinarily"]
negation_words = ["not", "never", "no", "none", "neither", "nor", "nowhere", "nothing", "no one", "nobody", "nevertheless", "without"]





# -------------Loading data--------------

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

        #Stores sentence ID as keys and also initializes the prediction nested dictionary.
        if aspectTerms:
            dataset[sentence_id] = {
                'text': text,
                'aspect_terms': aspectTerms,
                'predictions': {}
            }
            

    return dataset



#--------Rules for syntactic parsing----------


# Rule:1 "amod" (adjectival modifier) or "advmod" (adverbial modifier)
def analyze_sentiment_rule1(text, aspect_terms):
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

            # # Analyze sentiment based on the token's parent
            # parent = token.head
            # # Rule: If the aspect term's parent is a positive word from the opinion_lexicon
            # # with dependency type "amod", then the sentiment towards the aspect term is positive.
            # if (token.head.text.lower() in positive_words) and (token.dep_ == "amod" or token.dep_ == "advmod"):
            #     sentiment_scores[aspect_term] += 1

            # elif (token.head.text.lower() in negative_words) and (token.dep_ == "amod" or token.dep_ == "advmod"):
            #     sentiment_scores[aspect_term] -= 1

    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities

    

#Rule:2

def analyze_sentiment_rule2(text, aspect_terms):
    doc = nlp(text)

    sentiment_scores = {term: 0 for term in aspect_terms}

    for token in doc:
        if token.text in aspect_terms:
            aspect_term = token.text

            # Rule 1: If the aspect term is preceded by "not" or "never", then the sentiment is negative.
            if any(prev_token.text.lower() in ["not", "never"] for prev_token in token.lefts):
                sentiment_scores[aspect_term] -= 1

            # Rule 2: If the aspect term is preceded by "very" or "extremely", then the sentiment is positive.
            if any(prev_token.text.lower() in ["very", "extremely"] for prev_token in token.lefts):
                sentiment_scores[aspect_term] += 1

            # Rule 3: If the aspect term is followed by intensifiers like "very" or "extremely", then the sentiment is positive.
            if any(next_token.text.lower() in ["very", "extremely"] for next_token in token.rights):
                sentiment_scores[aspect_term] += 1

            # Rule 4: If the aspect term is followed by negation words like "not" or "never", then the sentiment is negative.
            if any(next_token.text.lower() in ["not", "never"] for next_token in token.rights):
                sentiment_scores[aspect_term] -= 1

            # Rule 5: If the aspect term is in a clause with a positive sentiment verb, then the sentiment is positive.
            if any(child.text.lower() in ["like", "love", "enjoy"] for child in token.children):
                sentiment_scores[aspect_term] += 1

            # Rule 6: If the aspect term is in a clause with a negative sentiment verb, then the sentiment is negative.
            if any(child.text.lower() in ["dislike", "hate"] for child in token.children):
                sentiment_scores[aspect_term] -= 1

            # Rule 7: If the aspect term is a superlative adjective, then the sentiment is positive.
            if token.tag_ == "JJS":
                sentiment_scores[aspect_term] += 1


    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative" 
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities
#---------------Precision and Recall Calculation--------------
    
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

rule = 1
# Analyze sentiments and store predictions
for sentence_id, data in dataset.items():
    text = data['text']
    aspect_terms = [term['term'] for term in data['aspect_terms']]
    if rule == 0:
        predictions = analyze_sentiment(text, aspect_terms)
    elif rule == 1:
        predictions = analyze_sentiment_rule1(text, aspect_terms)
    elif rule == 2:
        predictions = analyze_sentiment_rule2(text, aspect_terms)        
    data['predictions'] = predictions

precision, recall = calculate_precision_recall(dataset)


if rule == 0:
    print("Combination of all rules")
elif rule == 1:
    print("Rule:1 ""amod"" or ""advmod"" (considering parent and child)")
    
elif rule == 2:
    print("Rule:2 ") 

print("Precision:")
print("Positive:", precision['positive'])
print("Negative:", precision['negative'])
print("Neutral:", precision['neutral'])
print("")
print("Recall:")
print("Positive:", recall['positive'])
print("Negative:", recall['negative'])
print("Neutral:", recall['neutral'])