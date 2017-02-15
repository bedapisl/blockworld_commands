
from database import Database
import numpy as np
import pdb

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def vocabulary_length(vocabulary_version):
    db = Database()
    length = db.get_all_rows_single_element("SELECT Max(TokenID) FROM Vocabulary WHERE Version = " + str(vocabulary_version))[0] + 1

    return length


def get_embedding_matrix(vocabulary_version):
    embeddings_dict = {}
    f = open("../data/glove.6B.50d.txt", "r")
    for line in f.readlines():
        words = line.split()
        embeddings_dict[words[0]] = []
        for word in words[1:]:
            embeddings_dict[words[0]].append(float(word))
    

    db = Database()
    
    vocabulary_words = db.get_all_rows_single_element("SELECT Token FROM Vocabulary WHERE Version = " + str(vocabulary_version) + " ORDER BY TokenID ASC")

    embeddings = []

    missing = []
    for word in vocabulary_words:
        if word in embeddings_dict.keys():
            embeddings.append(embeddings_dict[word])
        else:
            missing.append(word)
            embeddings.append(np.random.uniform(-1.0, 1.0, size = 50))

    return np.array(embeddings)
        

def get_word_characters(vocabulary_version):
    db = Database()

    end_of_word_char = '$'

    words = db.get_all_rows_single_element("SELECT Token FROM Vocabulary WHERE Version = " + str(vocabulary_version) + " ORDER BY TokenID ASC")

    word_lens = [len(word) for word in words]
    
    all_letters = ""
    for word in words:
        all_letters += word

    all_letters = list(set(all_letters))
    all_letters.append(end_of_word_char)
        
    matrix = []
    for word in words:
        matrix.append([])
        for letter in word:
            matrix[-1].append(all_letters.index(letter))
        
        while len(matrix[-1]) < max(word_lens):
            matrix[-1].append(all_letters.index(end_of_word_char))

    return (matrix, len(all_letters), word_lens)


