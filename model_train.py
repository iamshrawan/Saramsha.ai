from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize, sent_tokenize

training_corpus = open('news_tokens.txt',encoding='utf-8')
training_data = training_corpus.read()
training_data = training_data.split('\n')

print(training_data[0:3])

# Tag the sentences
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
               for i, _d in enumerate(training_data)]

print(tagged_data[0:3])


max_epochs = 20
vec_size = 300
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save('oknews.model')
