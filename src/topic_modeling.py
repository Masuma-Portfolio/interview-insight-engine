from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def perform_topic_modeling(texts, num_topics=2):
    stop_words = set(stopwords.words('english'))
    tokenized = [[word for word in word_tokenize(doc.lower()) if word.isalpha() and word not in stop_words] for doc in texts]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model.print_topics()
