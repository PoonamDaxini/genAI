from nltk.stem import PorterStemmer, SnowballStemmer

stemmer = PorterStemmer()
print(stemmer.stem('running'))

stemmer1 = SnowballStemmer('english')
print(stemmer1.stem('going'))