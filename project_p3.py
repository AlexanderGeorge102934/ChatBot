from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk
from time import localtime, strftime
from nltk.parse.corenlp import CoreNLPDependencyParser

#-----------------------------------FILE WRITER-------------------------------------------------------

f = open("{0}.txt".format(strftime("%Y-%m-%d_%H:%M:%S", localtime())), "w")


# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    # Initialize NLTK tokenizer
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK tokenizer not found, downloading...")
        nltk.download('punkt')
    return nltk.tokenize.word_tokenize(inp_str)


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix

    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True, token_pattern=None)
    X_train_tfidf = vectorizer.fit_transform(training_documents)
    return vectorizer, X_train_tfidf


    return vectorizer, tfidf_train


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300, )

    if token in word2vec:
        word_vector = word2vec[token]

    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    tokens = get_tokens(user_input)
    embeddings = []
    for token in tokens:
        token_embedding = w2v(word2vec, token)
        embeddings.append(token_embedding)

    if not embeddings:
        return np.zeros(300)
    averaged_embedding = np.mean(embeddings, axis=0)

    return averaged_embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Four instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = GaussianNB()
    lr = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return nb, lr, svm, mlp


# Function: train_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    tfidf_dense = tfidf_train.toarray()
    model.fit(tfidf_dense, training_labels)
    return model


# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    embeddings_list = []

    for document in training_documents:
        doc_embedding = string2vec(word2vec, document)
        embeddings_list.append(doc_embedding)

    X_train_w2v = np.array(embeddings_list)
    model.fit(X_train_w2v, training_labels)
    return model


# Function: test_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    X_test_tfidf = vectorizer.transform(test_documents).toarray()
    predicted_labels = model.predict(X_test_tfidf)

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_w2v(model, word2vec, test_documents, test_labels):
    test_embeddings = []

    for doc in test_documents:
        doc_embedding = string2vec(word2vec, doc)
        test_embeddings.append(doc_embedding)

    X_test_w2v = np.array(test_embeddings)
    predicted_labels = model.predict(X_test_w2v)

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy


# Function: compute_ttr(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the type-token ratio for tokens in the input string.
# Type-token ratio is computed as: num_types / num_tokens, where num_types is
# the number of unique tokens.
def compute_ttr(user_input):
    # Tokenize the input text
    tokens = nltk.tokenize.word_tokenize(user_input)

    # Create a set of unique tokens
    unique_tokens = set(tokens)

    # Calculate the Type-Token Ratio (TTR)
    if len(tokens) > 0:
        ttr = len(unique_tokens) / len(tokens)
    else:
        ttr = 0

    # Return the TTR value
    return ttr


# Function: tokens_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of tokens per sentence
def tokens_per_sentence(user_input):
    # Tokenize the input into sentences
    sentences = nltk.tokenize.sent_tokenize(user_input)

    # Count tokens in each sentence
    token_counts = []
    for sentence in sentences:
        token_count = len(nltk.tokenize.word_tokenize(sentence))
        token_counts.append(token_count)

    # Calculate the average tokens per sentence (TPS)
    if len(sentences) > 0:
        tps = sum(token_counts) / len(sentences)
    else:
        tps = 0

    # Return the TPS value
    return tps


# Function: get_dependency_parse(input)
# This function accepts a raw string input and returns a CoNLL-formatted output
# string with each line indicating a word, its POS tag, the index of its head
# word, and its relation to the head word.
# Parameters:
# input - A string containing a single text input (e.g., a sentence).
# Returns:
# output - A string containing one row per word, with each row containing the
#          word, its POS tag, the index of its head word, and its relation to
#          the head word.
def get_dependency_parse(input: str):
    # Initialize the CoreNLPDependencyParser with the server URL
    parser = CoreNLPDependencyParser(url='http://localhost:9000')

    parse = parser.raw_parse(input)

    dep_graph = next(parse)

    output = dep_graph.to_conll(4)

    return output


# Function: get_dep_categories(parsed_input)
# parsed_input: A CONLL-formatted string.
# Returns: Five integers, corresponding to the number of nominal subjects (nsubj),
#          direct objects (obj), indirect objects (iobj), nominal modifiers (nmod),
#          and adjectival modifiers (amod) in the input, respectively.
#
# This function counts the number of grammatical relations belonging to each of five
# universal dependency relation categories specified for the provided input.
def get_dep_categories(parsed_input):
    num_nsubj = 0
    num_obj = 0
    num_iobj = 0
    num_nmod = 0
    num_amod = 0

    # Split the parsed_input by lines
    lines = parsed_input.strip().splitlines()

    # Iterate over each line and check for specific dependencies
    for line in lines:
        parts = line.split()  # Split each line by whitespace to access relation type
        if len(parts) >= 4:  # Ensure there are enough columns
            relation = parts[3]  # The fourth column should be the relation type

            # Increment counts based on the relation type
            if relation == 'nsubj':
                num_nsubj += 1
            elif relation == 'obj':
                num_obj += 1
            elif relation == 'iobj':
                num_iobj += 1
            elif relation == 'nmod':
                num_nmod += 1
            elif relation == 'amod':
                num_amod += 1

    return num_nsubj, num_obj, num_iobj, num_nmod, num_amod


# Function: custom_feature_1(user_input)
# user_input: A string of arbitrary length
# Returns: An output specific to the feature type implemented.
#
# This function implements a custom stylistic feature extractor.
def custom_feature_1(user_input):
    sentences = re.split(r'[.!?]', user_input)

    # Step 2: Initialize a list to hold word counts for each sentence
    word_counts = []
    for sentence in sentences:
        # Strip whitespace and check if the sentence is not empty
        if sentence.strip():
            # Count the number of words in the sentence
            word_count = len(sentence.split())
            word_counts.append(word_count)

    # Step 3: Calculate the average sentence length
    if len(word_counts) > 0:
        avg_sentence_length = sum(word_counts) / len(word_counts)
    else:
        avg_sentence_length = 0

    # Step 4: Return the average sentence length
    return avg_sentence_length


# Function: custom_feature_2(user_input)
# user_input: A string of arbitrary length
# Returns: An output specific to the feature type implemented.
#
# This function implements a custom stylistic feature extractor.
def custom_feature_2(user_input):
    total_chars = len(user_input)

    # Step 2: Initialize a counter for punctuation characters
    punctuation_count = 0
    for char in user_input:
        if char in string.punctuation:
            punctuation_count += 1

    # Step 3: Calculate punctuation density
    if total_chars > 0:
        punctuation_density = punctuation_count / total_chars
    else:
        punctuation_density = 0

    # Step 4: Return the punctuation density
    return punctuation_density



# Function: welcome_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    user_input = print("Welcome to my chatbot!  ")
    f.write("CHATBOT:\nWelcome to my chatbot!  ")

    return "get_user_info"


# Function: get_info_state()
# This function does not take any input
# Returns: A string indicating the next state and a string indicating the
#          user's name
#
# This function implements a state that requests the user's name and then processes
# the user's response to extract that information.  Feel free to customize this!
def get_info_state():
    # Request the user's name, and accept a user response of
    # arbitrary length.  Feel free to customize this!
    user_input = input("What is your name?\n")
    f.write("What is your name?\n")
    f.write("\nUSER:\n{0}\n".format(user_input))

    # Extract the user's name
    name = extract_user_info(user_input)

    return "sentiment_analysis", name


# Function: sentiment_analysis_state(name, model, vectorizer, word2vec)
# name: A string indicating the user's name
# model: The trained classification model used for predicting sentiment
# vectorizer: OPTIONAL; The trained vectorizer, if using TFIDF (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using Word2Vec (leave empty otherwise)
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what they want to talk about,
# and then processes their response to predict their current sentiment.  Feel free
# to customize this!
def sentiment_analysis_state(name, model, vectorizer=None, word2vec=None):
    # Check the user's sentiment
    user_input = input("Thanks {0}!  What do you want to talk about today?\n".format(name))
    f.write("\nCHATBOT:\nThanks {0}!  What do you want to talk about today?\n".format(name))
    f.write("\nUSER:\n{0}\n".format(user_input))

    # Predict the user's sentiment
    # test = vectorizer.transform([user_input])  # Use if you selected a TFIDF model
    test = string2vec(word2vec, user_input)  # Use if you selected a w2v model

    label = None
    label = model.predict(test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
        f.write("\nCHATBOT:\nHmm, it seems like you're feeling a bit down.\n")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
        f.write("\nCHATBOT:\nIt sounds like you're in a positive mood!\n")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
        f.write("\nCHATBOT:\nHmm, that's weird.  My classifier predicted a value of: {0}\n".format(label))

    return "stylistic_analysis"


# Function: stylistic_analysis_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response.  Feel free to customize this!
def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    f.write("\nCHATBOT:\nI'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    f.write("\nUSER:\n{0}\n".format(user_input))
    ttr = compute_ttr(user_input)
    tps = tokens_per_sentence(user_input)
    dep_parse = get_dependency_parse(user_input)
    num_nsubj, num_obj, num_iobj, num_nmod, num_amod = get_dep_categories(dep_parse)
    custom_1 = custom_feature_1(user_input)
    custom_2 = custom_feature_2(user_input)

    # Generate a stylistic analysis of the user's input
    print("Thanks!  Here's what I discovered about your writing style.")
    print("Type-Token Ratio: {0}".format(ttr))
    print("Average Tokens Per Sentence: {0}".format(tps))
    # print("Dependencies:\n{0}".format(dep_parse)) # Uncomment to view the full dependency parse.
    print("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}".format(num_nsubj, num_obj,
                                                                           num_iobj, num_nmod, num_amod))
    print("Custom Feature #1: {0}".format(custom_1))
    print("Custom Feature #2: {0}".format(custom_2))

    f.write("\nCHATBOT:\nThanks!  Here's what I discovered about your writing style.\n")
    f.write("Type-Token Ratio: {0}\n".format(ttr))
    f.write("Average Tokens Per Sentence: {0}\n".format(tps))
    # f.write("Dependencies:\n{0}\n".format(dep_parse)) # Uncomment to view the full dependency parse.
    f.write("# Nominal Subjects: {0}\n# Direct Objects: {1}\n# Indirect Objects: {2}"
          "\n# Nominal Modifiers: {3}\n# Adjectival Modifiers: {4}\n".format(num_nsubj, num_obj,
                                                                           num_iobj, num_nmod, num_amod))
    f.write("Custom Feature #1: {0}\n".format(custom_1))
    f.write("Custom Feature #2: {0}\n".format(custom_2))

    return "check_next_action"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user can indicate that they would like to quit, redo the sentiment
# analysis, or redo the stylistic analysis.  Feel free to customize this!
def check_next_state():
    next_state = ""

    user_input = input("What would you like to do next?  You can quit, redo the "
                       "sentiment analysis, or redo the stylistic analysis.\n")
    f.write("\nCHATBOT:\nWhat would you like to do next?  You can quit, redo the "
                       "sentiment analysis, or redo the stylistic analysis.\n")
    f.write("\nUSER:\n{0}\n".format(user_input))

    match = False # Only becomes true once a match is found
    while not match:
        quit_match = re.search(r"\bquit\b", user_input) # This is *not* comprehensive ...feel free to update! -Natalie
        sentiment_match = re.search(r"\bsentiment\b", user_input)
        analysis_match = re.search(r"\bstyl", user_input)

        if quit_match is not None:
            next_state = "quit"
            match = True
        elif sentiment_match is not None:
            next_state = "sentiment_analysis"
            match = True
        elif analysis_match is not None:
            next_state = "stylistic_analysis"
            match = True
        else:
            user_input = input("Sorry, I didn't understand that.  Would you like "
                               "to quit, redo the sentiment analysis, or redo the stylistic analysis?\n")
            f.write("\nCHATBOT:\nSorry, I didn't understand that.  Would you like "
                               "to quit, redo the sentiment analysis, or redo the stylistic analysis?\n")
            f.write("\nUSER:\n{0}\n".format(user_input))

    return next_state


# Function: run_chatbot(model, vectorizer=None):
# model: A trained classification model
# vectorizer: OPTIONAL; The trained vectorizer, if using Naive Bayes (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system---it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, vectorizer=None, word2vec=None):
    """
        Implements the main chatbot system using finite state dialogue management.

        Parameters:
        - model: A trained classification model for sentiment analysis.
        - vectorizer: (Optional) The trained TfidfVectorizer model.
        - word2vec: (Optional) The pretrained Word2Vec model.

        Returns:
        - None
        """
    state = "welcome_state"  # Initial state
    name = ""  # To store the user's name
    sentiment_analysis_run = 0  # Counter for sentiment analysis runs

    while state != "quit":
        if state == "welcome_state":
            # Call the welcome_state function
            state = welcome_state()

        elif state == "get_user_info":
            # Call the get_info_state function and retrieve the next state and name
            state, name = get_info_state()

        elif state == "sentiment_analysis":
            # Perform sentiment analysis
            state = sentiment_analysis_state(name, model, vectorizer, word2vec)
            sentiment_analysis_run += 1

            # Decide the next state based on the number of times sentiment analysis has run
            if sentiment_analysis_run == 1:
                state = "stylistic_analysis"
            else:
                state = "check_next_state"

        elif state == "stylistic_analysis":
            # Perform stylistic analysis
            state = stylistic_analysis_state()
            state = "check_next_state"

        elif state == "check_next_state":
            # Determine the next action based on user input
            state = check_next_state()

        else:
            # Handle unexpected states
            print(f"Unknown state encountered: {state}")
            break

    print("Chatbot session has ended. Goodbye!")

    return


# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.
if __name__ == "__main__":
    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)  # Use if you selected a Word2Vec model

    # Compute TFIDF representations so that you can make use of them later
    # vectorizer, tfidf_train = vectorize_train(documents)  # Use if you selected a TFIDF model

    # Instantiate and train the machine learning models
    # To save time, only uncomment the lines corresponding to the sentiment
    # analysis model you chose for your chatbot!

    # nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models() # Uncomment to instantiate a TFIDF model
    nb_w2v, logistic_w2v, svm_w2v, mlp_w2v = instantiate_models()  # Uncomment to instantiate a w2v model
    # nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    # nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    # logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    # logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    # svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    svm_w2v = train_model_w2v(svm_w2v, word2vec, documents, labels)
    # mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    # mlp_w2v = train_model_w2v(mlp_w2v, word2vec, documents, labels)

    # ***** New in Project Part 3! *****
    # next_state = welcome_state() # Uncomment to check how this works
    # next_state, name = get_info_state() # Uncomment to check how this works
    # next_state = sentiment_analysis_state(name, svm_w2v, word2vec=word2v) # Uncomment to check how this works---note that you'll need to update parameters to use different sentiment analysis models!
    # next_state = stylistic_analysis_state() # Uncomment to check how this works
    # next_state = check_next_state() # Uncomment to check how this works

    # run_chatbot(mlp, word2vec=word2vec) # Example for running the chatbot with
                                        # MLP (make sure to comment/uncomment
                                        # properties of other functions as needed)
    run_chatbot(svm_w2v, word2vec=word2vec) # Example for running the chatbot with SVM and Word2Vec---make sure your earlier functions are copied over for this to work correctly!
    f.close()
    