import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample Kerala-related news text
text = ("A serious environmental threat looms over the Arabian Sea off the Kerala "  
        "coast due to a fire aboard the Singapore-flagged vessel MV Wan Hai 503, "
        "which is carrying a large quantity of hazardous chemicals.")

# Process the text
doc = nlp(text)

# Tokenization
tokens = [token.text for token in doc]
print("1. Tokens:")
print(tokens)

# Stop-word removal
non_stop_tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
print("\n2. After Stop-word Removal:")
print(non_stop_tokens)

# Lemmatization (excluding stopwords & punctuation)
lemmas = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
print("\n3. Lemmatized Words:")
print(lemmas)


# token.text	Raw tokens	['A', 'serious', 'environmental', ...]
# token.is_stop	Identifies non-informative words	Removes "a", "is", "the", etc.
# token.is_alpha	Keeps only actual words (no symbols or digits)	Removes "503", ","
# token.lemma_	Lemmatizes to base form	"carrying" → "carry"