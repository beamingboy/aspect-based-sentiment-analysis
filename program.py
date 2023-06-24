# Student ID: a1802961
# Name      : Vinay Kumar


# Library imports.....
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
def analyze_sentiment_amod(text, aspect_terms):
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
                if child.text.lower() in positive_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] += 1

                # Rule: If the aspect term's child is a negative word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is negative.
                elif child.text.lower() in negative_words and child.dep_ == "amod":
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


# this uses nsubj
def analyze_sentiment_subject(text, aspect_terms):
    doc = nlp(text)

    # Initialize sentiment scores for each aspect term
    sentiment_scores = {term: 0 for term in aspect_terms}

    for token in doc:
        # Check if the token is an aspect term
        if token.text in aspect_terms:
            aspect_term = token.text

            # Analyze sentiment based on the token's children and dependencies
            for child in token.children:
               # Rule: If the aspect term's child has dependency type "nsubj"
                # and matches certain positive sentiment words from the opinion lexicon,
                # then the sentiment towards the aspect term is positive.
                if child.dep_ == "nsubj" and child.text.lower() in positive_words:
                    sentiment_scores[aspect_term] += 1

                # Rule: If the aspect term's child has dependency type "nsubj"
                # and matches certain negative sentiment words from the opinion lexicon,
                # then the sentiment towards the aspect term is negative.
                elif child.dep_ == "nsubj" and child.text.lower() in negative_words:
                    sentiment_scores[aspect_term] -= 1

                # Rule: If the aspect term's child has dependency type "nsubj"
                # and does not match any positive or negative sentiment words,
                # then the sentiment towards the aspect term is neutral.
                elif child.dep_ == "nsubj":
                    sentiment_scores[aspect_term] += 0
    # Determine sentiment polarities based on the sentiment scores
    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities


#Adjectival Modifiers (amod) + Noun Subject (nsubj)

#Rule: If the aspect term's child is an adjectival modifier (amod) and its grandchild is a noun subject (nsubj), 
# then the sentiment towards the aspect term is considered positive or negative based on 
# the sentiment of the adjectival modifier.

def analyze_sentiment_amod_nsubj(text, aspect_terms):
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
                if child.text.lower() in positive_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] += 1

                # Rule: If the aspect term's child is a negative word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is negative.
                elif child.text.lower() in negative_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] -= 1

                # Rule: If the aspect term's child has dependency type "amod" but is not a positive or negative word,
                # then check if the grandchild is a noun subject (nsubj) and assign sentiment accordingly.
                elif child.dep_ == "amod":
                    for grandchild in child.children:
                        if grandchild.dep_ == "nsubj" and grandchild.pos_ == "NOUN":
                            if child.text.lower() in positive_words:
                                sentiment_scores[aspect_term] += 1
                            elif child.text.lower() in negative_words:
                                sentiment_scores[aspect_term] -= 1

    # Determine sentiment polarities based on the sentiment scores
    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities






def analyze_sentiment_last(text, aspect_terms):
    doc = nlp(text)

    # Initialize sentiment scores for each aspect term
    sentiment_scores = {term: 0 for term in aspect_terms}

    for token in doc:
        # Check if the token is an aspect term
        if token.text in aspect_terms:
            aspect_term = token.text

            # Analyze sentiment based on the token's children and dependencies
            for child in token.children:
                # Rule: If the aspect term's child is an adverb and has dependency type "advmod",
                # then it modifies the sentiment of the aspect term.
                if child.pos_ == "ADV" and child.dep_ == "advmod":
                    modifier = child.text.lower()

                    # Increment or decrement sentiment score based on the adverb's sentiment modifier
                    if modifier in positive_words:
                        sentiment_scores[aspect_term] += 1
                    elif modifier in negative_words:
                        sentiment_scores[aspect_term] -= 1

                # Rule: If the aspect term's child is a positive or negative word from the opinion_lexicon
                # with dependency type "amod", then the sentiment towards the aspect term is modified accordingly.
                elif child.text in positive_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] += 1
                elif child.text in negative_words and child.dep_ == "amod":
                    sentiment_scores[aspect_term] -= 1

                # Rule: If the aspect term's child has dependency type "amod" but is not a positive or negative word,
                # then the sentiment towards the aspect term is neutral.
                elif child.dep_ == "amod":
                    sentiment_scores[aspect_term] += 0

            # Check for negation in the sentence and reverse sentiment if applicable
            for ancestor in token.ancestors:
                if ancestor.text.lower() in negative_words:
                    sentiment_scores[aspect_term] *= -1

    # Determine sentiment polarities based on the sentiment scores
    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities


def analyze_sentiment_nsubj_cop(text, aspect_terms):
    doc = nlp(text)

    # Initialize sentiment scores for each aspect term
    sentiment_scores = {term: 0 for term in aspect_terms}

    for token in doc:
        # Check if the token is an aspect term
        if token.text in aspect_terms:
            aspect_term = token.text

            # Analyze sentiment based on the token's children and dependencies
            for child in token.children:
                # Rule: If the aspect term's child has dependency type "nsubj" and the grandchild is a copular verb (cop)
                # followed by an adjective (amod), then assign sentiment accordingly.
                if child.dep_ == "nsubj" and child.pos_ == "NOUN":
                    for grandchild in child.children:
                        if grandchild.dep_ == "cop" and grandchild.pos_ == "AUX":
                            for adj in grandchild.children:
                                if adj.dep_ == "amod" and adj.pos_ == "ADJ":
                                    if adj.text.lower() in positive_words:
                                        sentiment_scores[aspect_term] += 1
                                    elif adj.text.lower() in negative_words:
                                        sentiment_scores[aspect_term] -= 1

    # Determine sentiment polarities based on the sentiment scores
    sentiment_polarities = {
        term: "positive" if score > 0 else "neutral" if score == 0 else "negative"
        for term, score in sentiment_scores.items()
    }

    return sentiment_polarities
#-------------Precision calculator-----------

# Function to store predicted polarities and calculate precision and recall
def evaluate_polarities(dataset, rule):
    total_positive = 0
    total_negative = 0
    total_neutral = 0
    true_positive_positive = 0
    true_positive_negative = 0
    true_positive_neutral = 0
    detected_positive = 0
    detected_negative = 0
    detected_neutral = 0

    for sentence_id, data in dataset.items():
        aspect_terms = [aspect_term['term'] for aspect_term in data['aspect_terms']]
        

        #rule used
        if rule == 1:
            predicted_polarities = analyze_sentiment_amod(data['text'], aspect_terms)
        elif rule == 2:
            predicted_polarities = analyze_sentiment_subject(data['text'], aspect_terms)
        elif rule == 3:
            predicted_polarities = analyze_sentiment_amod_nsubj(data['text'], aspect_terms)
        elif rule == 4:
            predicted_polarities = analyze_sentiment_rules(data['text'], aspect_terms)

        for aspect_term in aspect_terms:
            ground_truth_polarity = data['aspect_terms'][aspect_terms.index(aspect_term)]['polarity']
            predicted_polarity = predicted_polarities[aspect_term]

            if ground_truth_polarity == 'positive':
                total_positive += 1
                if predicted_polarity == 'positive':
                    true_positive_positive += 1
                if predicted_polarity != 'neutral':
                    detected_positive += 1

            elif ground_truth_polarity == 'negative':
                total_negative += 1
                if predicted_polarity == 'negative':
                    true_positive_negative += 1
                if predicted_polarity != 'neutral':
                    detected_negative += 1

            elif ground_truth_polarity == 'neutral':
                total_neutral += 1
                if predicted_polarity == 'neutral':
                    true_positive_neutral += 1
                elif predicted_polarity != 'neutral':
                    detected_neutral += 1

            data['predicted_polarities'] = predicted_polarities

    precision_positive = true_positive_positive / detected_positive * 100 if detected_positive > 0 else 0
    precision_negative = true_positive_negative / detected_negative * 100 if detected_negative > 0 else 0
    precision_neutral = true_positive_neutral / detected_neutral * 100 if detected_neutral > 0 else 0

    recall_positive = true_positive_positive / total_positive * 100 if total_positive > 0 else 0
    recall_negative = true_positive_negative / total_negative * 100 if total_negative > 0 else 0
    recall_neutral = true_positive_neutral / total_neutral * 100 if total_neutral > 0 else 0

    precision = {
        'positive': precision_positive,
        'negative': precision_negative,
        'neutral': precision_neutral
    }

    recall = {
        'positive': recall_positive,
        'negative': recall_negative,
        'neutral': recall_neutral
    }

    return dataset, precision, recall

# -------------Main--------------

# Usage example
file_path = 'Restaurants.xml'
dataset = load_data(file_path)

rule = 4
# Evaluate polarities and calculate precision and recall
dataset, precision, recall = evaluate_polarities(dataset,rule)

# Print precision and recall
print(f"Precision: {precision}")
print(f"Recall: {recall}")
# Print the loaded data








