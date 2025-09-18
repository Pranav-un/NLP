import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# 1. Kerala-related sample news text
text = ("A serious environmental threat looms over the Arabian Sea off the Kerala "
        "coast due to a fire aboard the Singapore-flagged vessel MV Wan Hai 503, "
        "which is carrying a large quantity of hazardous chemicals.")

# 2. Tokenization: break into words
tokens = word_tokenize(text)
print("1. Tokens:")
print(tokens)

# 3. Stop-word removal: remove common useless words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
print("\n2. After Stop-word Removal:")
print(filtered_tokens)

# 4. Stemming: reduce words to base root (may be crude)
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
print("\n3. Stemmed Words:")
print(stemmed_words)

# 5. Lemmatization: reduce to dictionary form (cleaner than stemming)
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("\n4. Lemmatized Words:")
print(lemmatized_words)
