import re
from langdetect import detect
from nltk.corpus import wordnet
from wordcloud import STOPWORDS
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from google_trans_new import google_translator
from gensim.models import KeyedVectors
import flair



## Initialize Word2Vec model
google_word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin' ,binary=True)

## Initialize flair text classification model
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

translator = google_translator()
stopwords = list(STOPWORDS)
ps = PorterStemmer()

def extract_translated_text(text):
    '''
    Extract translated English text from the reviews

    param str text: given review text
    :return str: translated text 
    '''
    start_indices = re.search(r"\b(translated by google)\b",text)
    end_indices = re.search(r"\b(original)\b",text)
    return text[start_indices.end()+1 : end_indices.start()-1]

def detect_language_and_translate(text):
    '''
    Check given text, if is already translated just extract translation
    else detect language and translate

    param str text: given review text:
    :return str: translated text
    '''

    if "translated by google" in text and "original" in text:
        return extract_translated_text(text)
  
    try:

        lang = detect(text)
        if lang != 'en':
            translate_text = translator.translate(text,lang_src=lang, lang_tgt='en')
        return translate_text

    except:
        return ''
  
    return text

def preprocess_text(comment, remove_stopwords = True):
    """
    Preprocess given review text
    """

    comment=comment.lower()
    #remove \n
    comment=re.sub("[^a-z0-9 '.$]"," ",comment)
    comment = re.sub(r"((.)\2{3,})",r"",comment)
    
    if remove_stopwords:
      comment = " ".join([word for word in comment.split() if len(word)>=2 and word not in stopwords])
    
    if comment.strip():
      return ' '.join(comment.split())
    
    return ''

def get_synoyoms(word):
    """
    Generate Synonyms for a given word
    """

    try:
        synonyms = [word]
  
        synsets = wordnet.synsets(word)

        # generate synonyms using wordnet corpus
        for syn in synsets: 
            for word in syn.lemmas(): 
                synonyms.append(" ".join(word.name().lower().split("_")))

            for word in syn.hypernyms():
                tokens = word.name().split('.')
                synonyms.append(re.sub(r"[^a-z]",""," ".join(tokens[0].lower().split("_"))))

        # use only the first six synonyms
        synonyms=synonyms[:6]
    except:
        pass

    # generate synonyms using google word2vec model
    if word in google_word2vec:
      similar_words = google_word2vec.most_similar(word)
      for word,score in similar_words:
        if score>=0.8:
          synonyms.append(re.sub(r"[^a-z]","",preprocess_text(" ".join(word.lower().split("_")))))

    return list(set(synonyms))


def extract_keywords(df, topics, topics_vs_keywords):
    """
    Extract keywords from reviews that are semantically similar to the topic names
    """

    for line in df["clean_review"].values:
        text = line.split()
        for word in text:
            ##
            if '$' in word:
                topics_vs_keywords["financial"].add(word)
                continue

            for topic in topics:
                topic = topic.lower()
                stemmed_word = ps.stem(word)

                ## calculate similarity between topic and keyword using google_word2vec
                if topic in google_word2vec and word in google_word2vec:
                    word_similarity = google_word2vec.similarity(topic,word)
                else:
                    word_similarity = 0.0
                
                ## calculate similarity between topic and stemmed keyword using google_word2vec
                if topic in google_word2vec and stemmed_word in google_word2vec:
                    stemmed_word_similarity = google_word2vec.similarity(topic,stemmed_word)
                else:
                    stemmed_word_similarity = 0.0

                if (stemmed_word_similarity>=0.65 or word_similarity>=0.65):
                    topics_vs_keywords[topic].add(word)
    
    return topics_vs_keywords

def extract_topics(review, keywords, topics, keywords_vs_topics):
    """
    Extract topics from reviews
    """
    
    sentences = re.split(r"[.|,]",review)

    extracted_topics = {}
    topics_vs_sentence_index = {}
    for idx in range(len(sentences)):
    
        tokens = sentences[idx].split()

        unigrams = nltk.ngrams(tokens, 1)
        bigrams = nltk.ngrams(tokens, 2)
        tokens = list(unigrams) + list(bigrams)

        for token in tokens:
            token = " ".join(token)

        if token in keywords:
            tags = keywords_vs_topics.get(token)

            for tag in tags:
                extracted_topics[tag] = extracted_topics.get(tag,0)+1
                #print(tag,'--->',token)
                topics_vs_sentence_index[idx] = topics_vs_sentence_index.get(idx,[])
                topics_vs_sentence_index[idx].append(tag)
          
    return list(extracted_topics.keys()), topics_vs_sentence_index

def get_keywords_vs_topics(topics_vs_keywords):

    keywords_vs_topics = dict()
    topics_vs_keywords_expanded = dict()
    for key ,values in topics_vs_keywords.items():
        for v in values:
            synonyms = get_synoyoms(v)
            for w in synonyms:
            
                topics_vs_keywords_expanded[w] = topics_vs_keywords_expanded.get(w,set())
                topics_vs_keywords_expanded[w].add(key)

                keywords_vs_topics[key] = keywords_vs_topics.get(key,set())
                keywords_vs_topics[key].add(w)
    
    return keywords_vs_topics, topics_vs_keywords_expanded

def predict_sentiment(document):
    
    sent_obj = flair.data.Sentence(document)
    flair_sentiment.predict(sent_obj)
  
    if sent_obj.labels:
        sentiment = str(sent_obj.labels[0])
        sentiment = re.sub(r"\((.*?)\)","",sentiment).strip()
        return sentiment.lower()
  
    return ''