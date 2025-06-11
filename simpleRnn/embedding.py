from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
import numpy as np

### sentences
sent=[  'the glass of milk',
     'the glass of juice',
     'the cup of tea',
    'I am a good boy',
     'I am a good developer',
     'understand the meaning of words',
     'your videos are good',]

# vocabulary size
voc_size=10000

# One-hot encode the sentences
one_hot_sentences = [one_hot(sentence, voc_size) for sentence in sent]


sent_length = 8
# Pad the sequences to ensure uniform length
padded_sentences = pad_sequences(one_hot_sentences, padding='pre', maxlen=sent_length)

# feature representation
dim=10

model = Sequential()
# Add an embedding layer
model.add(Embedding(voc_size, dim, input_length=sent_length))
# Add a SimpleRNN layer

model.compile(optimizer='adam', loss='mse')

print(model.summary())

# Print the padded sentences
print("Padded Sentences:")
for i, sentence in enumerate(padded_sentences):
    print(f"Sentence {i+1}: {sentence}")
    print(model.predict(sentence))
    # print()