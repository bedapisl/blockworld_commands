from nltk.corpus import wordnet
import pdb 



def synonyms(word):
    result = []
    for meaning in wordnet.synsets(word):
        print(meaning.pos())
        result += meaning.lemma_names()

    return list(set(result))


print(synonyms("place"))
print(synonyms("one"))
print(synonyms("block"))
print(synonyms("target"))
print(synonyms("space"))
print(synonyms("spaces"))
print(synonyms("to"))


pdb.set_trace()


  # Get All Synsets for 'dog'
# This is essentially all senses of the word in the db
print(wordnet.synsets('dog'))
#[Synset('dog.n.01'), Synset('frump.n.01'), Synset('dog.n.03'), 
# Synset('cad.n.01'), Synset('frank.n.02'),Synset('pawl.n.01'), 
# Synset('andiron.n.01'), Synset('chase.v.01')]

# Get the definition and usage for the first synset
print(wordnet.synset('dog.n.01').definition)
#'a member of the genus Canis (probably descended from the common 
#wolf) that has been domesticated by man since prehistoric times; 
#occurs in many breeds'
print(wordnet.synset('dog.n.01').examples)
#['the dog barked all night']

# Get antonyms for 'good'
#wordnet.synset('good.a.01').lemmas[0].antonyms()
#[Lemma('bad.a.01.bad')]

# Get synonyms for the first noun sense of 'dog'
print(wordnet.synset('dog.n.01').lemmas)
#[Lemma('dog.n.01.dog'), Lemma('dog.n.01.domestic_dog'), 
#Lemma('dog.n.01.Canis_familiaris')]

# Get synonyms for all senses of 'dog'
for synset in wordnet.synsets('dog'): 
    print(synset.lemmas)
#[Lemma('dog.n.01.dog'), Lemma('dog.n.01.domestic_dog'), 
#Lemma('dog.n.01.Canis_familiaris')]
#...
#[Lemma('frank.n.02.frank'), Lemma('frank.n.02.frankfurter'), 
#...


