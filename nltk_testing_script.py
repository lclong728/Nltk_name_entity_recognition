import nltk
from nltk.tag.stanford import StanfordNERTagger
import os

# set up java path to system environment
java_path = "C:\Program Files\Java\jdk1.8.0_191"
os.environ['JAVA_HOME'] = java_path

# standford core nlp library
jar = './standford_ner_tagger/stanford-ner.jar'

# trained models, ready for classify
model_eng = './standford_ner_tagger/ner-model-english.ser.gz'
model_french = './standford_ner_tagger/french_testing_model.gz'

# test cases
sentence_eng = u"Twenty miles east of Reno, Nev., " \
    "where packs of wild mustangs roam free through " \
    "the parched landscape, Tesla Giga Factory 1 " \
    "sprawls near Interstate 80."

sentence_french = u"La première Falcon Heavy de l'entreprise SpaceX, " \
    "la plus puissante fusée des Etats-Unis jamais " \
    "lancée depuis plus de quarante ans, devrait bien " \
    "emporter le roadster de l'entrepreneur américain, " \
    "mais sur une orbite bien différente. Elon Musk a le sens du spectacle."

# Prepare NER tagger with english model
ner_tagger = StanfordNERTagger(model_eng, jar, encoding='utf8')

# Tokenize: Split sentence into words
words = nltk.word_tokenize(sentence_eng)

# Run NER tagger on words
result = ner_tagger.tag(words)
print(result)
