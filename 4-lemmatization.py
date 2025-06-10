import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmetizer = WordNetLemmatizer()
print(lemmetizer.lemmatize('running'))
print(lemmetizer.lemmatize('goes'))
print(lemmetizer.lemmatize('better', pos='a'))  """adjective"""
print(lemmetizer.lemmatize('better', pos='v'))  // verb
print(lemmetizer.lemmatize('better', pos='n'))  // noun
print(lemmetizer.lemmatize('better', pos='r'))  // adverb
print(lemmetizer.lemmatize('better', pos='s'))  // adverb
