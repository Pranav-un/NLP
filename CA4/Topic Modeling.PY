import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

nltk.download('stopwords')
nltk.download('punkt')

# Sample corpus
texts = [
    "Machine learning is a field of artificial intelligence.",
    "Deep learning allows computational models that are composed of multiple processing layers.",
    "Natural language processing is a branch of AI.",
    "Topic modeling discovers abstract topics in a corpus."
]

stop_words = set(stopwords.words('english'))

# Preprocess
processed_texts = []
for doc in texts:
    tokens = word_tokenize(doc.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    processed_texts.append(tokens)

# Create dictionary & corpus
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# LDA Model
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=10)

# Print topics
for idx, topic in lda_model.print_topics():
    print("Topic: {} \nWords: {}".format(idx, topic))

# Visualization
vis = gensimvis.prepare(lda_model, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')
print("\nInteractive LDA visualization saved as lda_visualization.html")
