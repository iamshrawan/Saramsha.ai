from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from gensim.utils import lemmatize
import bs4 as bs
import urllib.request

model = Doc2Vec.load('oknews.model')

def get_summary(news_link = "http://english.onlinekhabar.com/will-try-to-endorse-medical-education-bill-on-friday-says-speaker.html"):
    # Getting news content
    news_source = urllib.request.urlopen(news_link).read()
    news_soup = bs.BeautifulSoup(news_source,'lxml')
    news_content = news_soup.find_all('div', class_ = 'oke-content-wrap clearfix')
    news_portion = news_content[0].find_all('p')
    news_para = [n.text for n in news_portion]
    news_para = ' '.join(news_para)
    news = news_para.split('\n\t')[0]

    # Get sentences
    news = news.split('\n')
    news = ' '.join(news)
    sentence_tk = sent_tokenize(news)
    print(sentence_tk)

    # Lemmatizing sentences (finding root word)
    tokenized = []
    i = 1

    for sentence in sentence_tk:
        print(i)
        lemmatized_out = [wd.decode('utf-8').split('/')[0] for wd in lemmatize(sentence)]
        lemmatized_out = ' '.join(lemmatized_out)
        tokenized.append(lemmatized_out)
        i = i + 1

    print(tokenized)
    print('\n\n')

    #News sentences clustering
    clustering_data = []
    for token in tokenized:
        vec = model.infer_vector(token)
        clustering_data.append(vec)

    data_length = len(clustering_data)
    n_clusters = int(np.floor(data_length/3))
    kmeans = KMeans(n_clusters=n_clusters, n_init = 1)
    kmeans = kmeans.fit(clustering_data)

    #Getting representative sentences
    avg = []
    for j in range(n_clusters):
       idx = np.where(kmeans.labels_ == j)[0]
       avg.append(np.mean(idx))
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, clustering_data)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([sentence_tk[closest[idx]] for idx in ordering])
    #print(summary + '\n\n')
    #print('Length of original text: ',len(sentence_tk))
    #print('Length of summary: ',len(sent_tokenize(summary)))
    return summary

if __name__ == '__main__':
    example = get_summary()
    print(example)
    

