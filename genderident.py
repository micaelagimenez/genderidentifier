#Load packages
from nltk.corpus import names
import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy 

#Feature extractor
def gender_features(word):
    return {'last_letter': word[-1]}

gender_features('Maria')

#Exploring female names
names.fileids()
names.words('female.txt')[:5]

#Building the classifier
labeled_names = ([(name, 'female') for name in names.words('female.txt')] + [(name, 'male') for name in names.words('male.txt')])
labeled_names[:5]

random.shuffle(labeled_names)
labeled_names[:5]

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
featuresets[:5]

#Split data into training (80%) and test (20%) set
train_set_size = round(len(featuresets) * .8)
train_set, test_set = featuresets[:train_set_size], featuresets[train_set_size:]

test_names = labeled_names[train_set_size:]
classifier = NaiveBayesClassifier.train(train_set)

#Testing the classifier
classifier.labels()

classifier.classify(gender_features('Marta'))
classifier.classify(gender_features('Alex'))
classifier.classify(gender_features('Sam'))
classifier.classify(gender_features('Mica'))
