import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize, TreebankWordTokenizer

corpus = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
"""

print(corpus)
## tokenize the corpus
documents = sent_tokenize(corpus)
print(type(documents))

for sentences in documents:
    print(sentences)

for sentences in documents:
    print(word_tokenize(sentences))
print(wordpunct_tokenize(corpus))

print()
tokenize = TreebankWordTokenizer()
print(tokenize.tokenize(corpus))
# print(tokens)
