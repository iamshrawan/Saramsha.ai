from nltk.tokenize import sent_tokenize
from gensim.utils import lemmatize
# Prepare data fot training

sentence_tk = []
with open('ok_news.txt', encoding='utf-8') as corpus:
    articles = corpus.read()
    articles = articles.split('\n')
    for article in articles:
        token = sent_tokenize(article)
        sentence_tk += token

print(sentence_tk[0:3])

tokenized = []
print(len(sentence_tk))
i = 1

for sentence in sentence_tk:
    print(i)
    lemmatized_out = [wd.decode('utf-8').split('/')[0] for wd in lemmatize(sentence)]
    lemmatized_out = ' '.join(lemmatized_out)
    tokenized.append(lemmatized_out)
    i = i + 1

print(tokenized[0:3])

with open('news_tokens.txt', 'a', encoding='utf-8') as tokens:
    for sentence in tokenized:
        tokens.write(sentence + '\n')

tokens.close()


