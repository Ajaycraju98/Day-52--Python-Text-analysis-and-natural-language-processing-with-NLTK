# Day-52--Python-Text-analysis-and-natural-language-processing-with-NLTK
This project explains about the Text analysis and natural language processing with NLTK
# pip install nltk
# Import required NLTK modules
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, FreqDist

# Download necessary NLTK data (run this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
# Input Text
text = """
Natural Language Toolkit, or NLTK, is a powerful Python library for text analysis and natural language processing. 
It provides tools for tokenization, stemming, lemmatization, and more. Let's explore its features.
"""
# 1. Sentence Tokenization
print("1. Sentence Tokenization:")
sentences = sent_tokenize(text)
print(sentences)

# 2. Word Tokenization
print("\n2. Word Tokenization:")
words = word_tokenize(text)

# 3. Stopwords Removal
print("\n3. Stopwords Removal:")
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word.lower() not in stop_words]
print(filtered_words)
# 4. Stemming
print("\n4. Stemming (Using Porter Stemmer):")
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in filtered_words]
print(stems)

# 5. Lemmatization
print("\n5. Lemmatization:")
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]
print(lemmas)

# 6. Part-of-Speech (POS) Tagging
print("\n6. Part-of-Speech Tagging:")
pos_tags = pos_tag(filtered_words)
print(pos_tags)

# 7. Frequency Distribution
print("\n7. Frequency Distribution:")
fdist = FreqDist(filtered_words)
print("Most Common Words:", fdist.most_common(5))

# Visualizing Frequency Distribution (Optional)
try:
    import matplotlib.pyplot as plt
    fdist.plot(10, title="Top 10 Word Frequency Distribution")
    plt.show()
except ImportError:
    print("Matplotlib not installed, skipping plot.")
