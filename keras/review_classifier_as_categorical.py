from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

word_index = imdb.get_word_index()

def convert_encoded_to_string(data):
    review = []
    for given_value in data:
        for word, value in word_index.items():
            if value + 3 == given_value:
                review.append(word)
    return ' '.join(review)

def convert_string_to_encoded(data):
    review = []
    for given_word in data.split():
        for word, value in word_index.items():
            if word == given_word:
                review.append(value + 3)
    return review

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.utils.np_utils import to_categorical
model.fit(x_train, to_categorical(y_train), epochs=4, batch_size=512)
results = model.evaluate(x_test, to_categorical(y_test))

model.predict(x_test)

custom_review1 = ('this is the best movie i have ever seen in my life '
                  'amazing roles played by the characters this totally '
                  'reminds me of the actual novel which is a must read '
                  'as well')

custom_review2 = ('this film was a garbage the story was meaningless '
                  'and felt bad it was so boring that i felt i was '
                  'being forced to watch this utter useless show')

custom_review3 = ('the movie was fantastic, extraordinary every moment'
                  'of the movie was worth it. it was very enjoyable')

custom_reviews = (custom_review1, custom_review2, custom_review3)
encoded_custom_reviews = np.array([convert_string_to_encoded(review)
                                   for review in custom_reviews])
vectorized_custom_reviews = vectorize_sequences(encoded_custom_reviews)

prediction = model.predict(vectorized_custom_reviews)
print(prediction)
