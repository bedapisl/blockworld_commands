from database import Database
import pdb
import re
import hunspell
from nltk.corpus import wordnet
from scipy.spatial import distance
from setting import digits, logos, all_tags
from utils import levenshtein_distance
from ufal.udpipe import *
from enum import Enum

import pickle #only for some analysis

class TokenID:
    BOS = 0
    EOS = 1
    UNK = 2


class Hunspell:
    def __init__(self):
        self.hunspell_checker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
        self.hunspell_checker.add(",")
        for logo in logos:
            for logo_part in logo.split():
                self.hunspell_checker.add(logo_part)

   
    def alternatives_hunspell(self, token):
        try:
            if not self.hunspell_checker.spell(token):
                result = []
                for suggestion in self.hunspell_checker.suggest(token): 
                    result.append(suggestion.decode().lower())
                return result

        except UnicodeDecodeError:
            pass

        except UnicodeEncodeError:
            pass

        return []


class Tokenizer:
    def __init__(self, engine, ignore_words, use_hunspell):
        self.engine = engine
        self.ignore_words = ignore_words
        self.changes = {}
        self.use_hunspell = use_hunspell

        if use_hunspell:
            self.hunspell_checker = Hunspell()
        
        if self.engine == "udpipe":
            self.model = Model.load("../data/english-ud-1.2-160523.udpipe")
            self.pipeline = Pipeline(self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

       
    def udpipe_process(self, command):
        processed = self.pipeline.process(command.lower())
        words = processed.split("\n")
        tokens = []

        for word in words:
            parts = word.split("\t")
            if len(parts) < 3:
                continue
            
            tokens.append(parts[1])

        return tokens


    def tokenize(self, command):
        if self.engine == "udpipe":
            tokens = self.udpipe_process(command)

            new_tokens = []
            for i, token in enumerate(tokens):
                token = re.sub("([0-9])([a-zA-Z])", "\g<1> \g<2>", token)
                token_parts = token.replace("-", " ").replace("'", " ").replace("’", " ").replace("hp.", "hp").split(" ")
                for token_part in token_parts:
                    if token_part != '':
                        new_tokens.append(token_part)

            tokens = new_tokens

        elif self.engine == "rule_based":
            command = command.lower() + " "
            command = command.replace(",", " , ").replace("'s", "").replace("’s", "").replace("\"", "").replace("(", "").replace(")", "").replace(";", " ").replace(". ", " . ").replace("'", "")
            command = re.sub("(\D)-(\D)", "\g<1> \g<2>", command)       #substitutes - for space if there are not numbers before and after -
            command = re.sub("([a-zA-Z])([0-9])", "\g<1> \g<2>", command)
            command = re.sub("([0-9])([a-zA-Z])", "\g<1> \g<2>", command)
            
            tokens = command.split()

        elif self.engine == "simple":
            command = command.lower() + " "
            command = re.sub("([a-z])([^a-z ]", "\g<1> \g<2>", command)
            tokens = command.split()
        

        if self.use_hunspell:               #if hunspell suggest two word alternative, split token to 2 tokens
            new_tokens = []
            for token in tokens:
                corrections = self.hunspell_checker.alternatives_hunspell(token)
                if len(corrections) > 0 and (" " in corrections[0]):
                    new_tokens += corrections[0].split(" ")
                else:
                    new_tokens.append(token)
            
            tokens = new_tokens
                   
        return tokens


class LemmatizerTagger:
    def __init__(self):
        self.model = Model.load("../data/english-ud-1.2-160523.udpipe")
        self.pipeline = Pipeline(self.model, "vertical", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    
    def lemmatize_tag(self, command):
        command_with_newlines = ""
        for token in command:
            command_with_newlines += token + "\n"
        processed = self.pipeline.process(command_with_newlines)          #TODO should here be lower??
        words = processed.split("\n")
        lemmas = []
        tags = []

        for word in words:
            parts = word.split("\t")
            if len(parts) < 3:
                continue
            
            lemmas.append(parts[2])
            tags.append(parts[3])

        return lemmas, tags


class SpellChecker:
    def __init__(self, vocabulary_version, tokenizer, lemmatizer, occurences_to_repair, use_hunspell, use_synonyms, use_lemma, max_levenshtein_distance, min_count, hunspell_first):
        self.occurences_to_repair = occurences_to_repair
        self.use_hunspell = use_hunspell
        self.use_lemma = use_lemma
        self.use_synonyms = use_synonyms
        self.max_levenshtein_distance = max_levenshtein_distance
        self.vocabulary_version = vocabulary_version
        self.min_count = min_count
        self.hunspell_first = hunspell_first
        self.all_tokens = None
        self.class_representatives = None
        self.tokens_list = None
        
        self.tokenizer = tokenizer
        self.lemmatizer = lemmatizer
        self.hunspell_checker = Hunspell()
    

    def load_token_list(self):
        db = Database()
        tokens = db.get_all_rows("SELECT TokenID, Token, Count FROM core.Vocabulary WHERE Version = " + str(self.vocabulary_version) + " ORDER BY Count DESC")
        self.tokens_list = []
        for (token_id, token, count) in tokens:
            self.tokens_list.append((token, count, {}, {}, token_id))


    def class_representative(self, token_class):
        if self.class_representatives is None:
            self.class_representatives = {}
            representatives_counts = {}
            for (token, count, _, _, token_class) in self.tokens_list:
                if not token_class in self.class_representatives:
                    self.class_representatives[token_class] = token
                    representatives_counts[token_class] = count

                elif representatives_counts[token_class] < count:
                    self.class_representatives[token_class] = token
                    representatives_counts[token_class] = count
        
        return self.class_representatives[token_class]
            

    def change_classes(self, old_class, new_class, tokens_list):
        new_tokens_list = []
        for token, count, lemmas, tags, token_class in tokens_list:
            if token_class == old_class:
                token_class = new_class

            new_tokens_list.append((token, count, lemmas, tags, token_class))

        return new_tokens_list
    

    def resolve_unknown(self, tokens_list):
        class_counts = {}
        for (_, count, _, _, token_class) in tokens_list:
            if token_class in class_counts:
                class_counts[token_class] += count
            else:
                class_counts[token_class] = count

        for token_class in class_counts.keys():
            if class_counts[token_class] < self.min_count:
                tokens_list = self.change_classes(token_class, int(TokenID.UNK), tokens_list)
        
        return tokens_list


    def compute_classes(self, tokens):
        self.tokens_list = []
        for i, key in enumerate(tokens.keys()):
            self.tokens_list.append((key, tokens[key][0], tokens[key][1], tokens[key][2], i + 3))

        self.tokens_list.sort(key = lambda tup: tup[1])

        for [token, count, lemmas, tags, token_class] in self.tokens_list:
            if count > self.occurences_to_repair:
                break
            
            correction = self.get_correct_class(token, lemmas, tags, use_min_count = False)
            self.tokens_list = self.change_classes(token_class, correction, self.tokens_list)

        self.tokens_list = self.resolve_unknown(self.tokens_list)
        
        return self.tokens_list


    def get_correct_class(self, token, lemmas, tags, use_min_count):
        if self.tokens_list is None:
            self.load_token_list()

        alternatives = [token]
        if self.use_hunspell:
            alternatives += self.hunspell_checker.alternatives_hunspell(token)

        if self.use_lemma:
            alternatives += list(lemmas)

        if self.use_synonyms:
            alternatives += self.alternatives_synonyms(token, list(tags))

        if self.max_levenshtein_distance > 0:
            if self.all_tokens == None:
                self.all_tokens = [token for (token, _, _, _, _) in self.tokens_list]

            alternatives += self.alternatives_levenshtein(token, self.all_tokens)

        alternatives = list(set(alternatives))

        if self.hunspell_first:
            alternatives = [token]

        for possible_correction in reversed(self.tokens_list):
            if possible_correction[0] in alternatives:
                return possible_correction[-1]
        
        return int(TokenID.UNK)


    def alternatives_synonyms(self, token, tags):
        wordnet_pos_tag = []

        for upostag in tags:
            if upostag in ["NOUN", "NUM"]:
                wordnet_pos_tag.append("n")

            elif upostag in ["VERB"]:
                wordnet_pos_tag.append("v")

            elif upostag in ["ADJ"]:
                wordnet_pos_tag.append("a")

            elif upostag in ["ADV"]:
                wordnet_pos_tag.append("r")


        result = []
        for meaning in wordnet.synsets(token):
            if meaning.pos() in wordnet_pos_tag:
                result += meaning.lemma_names()

        return list(set(result))


    def alternatives_levenshtein(self, token, all_tokens):
        result = []
        for other_token in all_tokens:
            if levenshtein_distance(other_token, token) <= self.max_levenshtein_distance:
                result.append(other_token)

        return result
    

    def create_vocabulary(self):
        db = Database()
        
        training_commands = db.get_all_rows_single_element("SELECT Command FROM Command JOIN Configuration ON Command.ConfigurationID = Configuration.ConfigurationID WHERE Configuration.Dataset = 'train'")
        
        tokens = {}
        for command in training_commands:
            command_tokens = self.tokenizer.tokenize(command)
            for i, token in enumerate(command_tokens):
                hunspell_alternatives = self.hunspell_checker.alternatives_hunspell(token)
                if len(hunspell_alternatives) > 0:
                    command_tokens[i] = hunspell_alternatives[0]

            lemmas, tags = self.lemmatizer.lemmatize_tag(command_tokens)
            for i, token in enumerate(command_tokens):
                if token in tokens:
                    tokens[token][0] += 1
                    tokens[token][1].update([lemmas[i]])
                    tokens[token][2].update([tags[i]])
                else:
                    tokens[token] = [1, set([lemmas[i]]), set([tags[i]])]

        token_classes = self.compute_classes(tokens)

        vocabulary_rows = []
        vocabulary_rows.append([int(TokenID.BOS), "<bos>", len(training_commands), self.vocabulary_version])
        vocabulary_rows.append([int(TokenID.EOS), "<eos>", len(training_commands), self.vocabulary_version])
        vocabulary_rows.append([int(TokenID.UNK), "<unk>", sum([count for (_, count, _, _, token_class) in token_classes if token_class == TokenID.UNK]), self.vocabulary_version])

        for (token, count, lemmas, tags, token_class) in token_classes:
            if token_class != int(TokenID.UNK):
                vocabulary_rows.append([token_class, token, count, self.vocabulary_version])

        db.execute("DELETE FROM Vocabulary WHERE Version = " + str(self.vocabulary_version))
        db.insert_many("Vocabulary", vocabulary_rows)
        
   
def get_world(configuration_id, start, finish, db):
    world_before = db.get_all_rows("SELECT X, Y, Z FROM World WHERE ConfigurationID = " + str(configuration_id) + " AND State = " + str(start) + " ORDER BY BlockID")
    world_after = db.get_all_rows("SELECT X, Y, Z FROM World WHERE ConfigurationID = " + str(configuration_id) + " AND State = " + str(finish) + " ORDER BY BlockID")
    return (world_before, world_after)


def get_changed_block(world_before, world_after):
    source = -1
    for block_id in range(0, len(world_before)):
        if world_before[block_id] != world_after[block_id]:
            if source != -1:
                print("Error: Multiple blocks changed in single step")
                pdb.set_trace()
            else:
                source = block_id
    
    return source


#returns block_id of block which can be referenced by the word
def get_possible_reference(word, decoration):
    result = -1
    if decoration == "digit":
        numerical = list(map(str, range(1,21)))
        if word in numerical:
            result = numerical.index(word)

        for i, digit_name in enumerate(digits):
            if levenshtein_distance(word, digit_name) < 2 and word != "none" and len(word) > 2:
                if result != -1:
                    print("Error: " + str(word) + " is ambiguous reference")
                    assert False
                result = i

    if decoration == "logo":
        if word == "hps":
            return logos.index("hp")
        if word == "stell":
            return logos.index("stella artois")

        for i, logo in enumerate(logos):
            for logo_part in logo.split():
                if (levenshtein_distance(word, logo_part) < 2 and len(word) > 2) or word == logo_part:
                    if result != -1 and result != i:
                        print("Warning: " + str(word) + " is ambiguous reference")
                    result = i

    return result


# returns nearest block which could have been referenced in command
def get_reference(command, decoration, location, source, world_after, tokenizer):
    possible_references = set(map(lambda token : get_possible_reference(token, decoration), tokenizer.tokenize(command)[0]))

    min_distance = 10000000
    best_reference = -1
    for possible_reference in possible_references:
        if possible_reference != -1 and possible_reference != source and possible_reference < len(world_after):  
            if distance.euclidean(location, world_after[possible_reference]) < min_distance:
                min_distance = distance.euclidean(location, world_after[possible_reference])
                best_reference = possible_reference

    return best_reference


#returns source minus reference
def get_direction(source, reference, world_after):
    return [s_i - r_i for s_i, r_i in zip(world_after[source], world_after[reference])]


def encode_command(command, tokenizer, lemmatizer, spell_checker):
    encoded = []
    encoded.append(int(TokenID.BOS))

    tokens = tokenizer.tokenize(command)

    lemmas, command_tags = lemmatizer.lemmatize_tag(tokens)
            
    for i, word in enumerate(tokens):
        token_class = spell_checker.get_correct_class(word, {lemmas[i]}, {command_tags[i]}, use_min_count = True)
        encoded.append(token_class)
        if token_class == int(TokenID.UNK):
            tokens[i] = "<unk>(" + word + ")"
        else:
            tokens[i] = spell_checker.class_representative(token_class)
    
    encoded.append(int(TokenID.EOS))

    if command_tags is not None:
        command_tags.insert(0, "X")
        command_tags.append("X")
        
        encoded_tags = []
        for tag in command_tags:
            encoded_tags.append(all_tags.index(tag))

    else:
        encoded_tags = None

    return encoded, encoded_tags, tokens
    

def create_training_data(version, tokenizer, lemmatizer, spell_checker):
    db = Database()
    all_commands = db.get_all_rows("SELECT Com.CommandID, Conf.Dataset, Conf.Decoration, Com.Command, Com.ConfigurationID, Com.Start, Com.Finish FROM Command AS Com JOIN Configuration AS Conf ON Com.ConfigurationID = Conf.ConfigurationID ORDER BY Com.CommandID")
    
    data = []
    i = 0
    for (command_id, dataset, decoration, command, configuration_id, start, finish) in all_commands:
        i += 1
        if i % 100 == 0:
            print(str(i) + "/" + str(len(all_commands)))

        (world_before, world_after) = get_world(configuration_id, start, finish, db)
        source = get_changed_block(world_before, world_after)
        location = world_after[source]
        reference = get_reference(command, decoration, location, source, world_after, tokenizer)
        direction = get_direction(source, reference, world_after)
        encoded_command, encoded_tags, tokens = encode_command(command, tokenizer, lemmatizer, spell_checker)
        
        if encoded_tags is not None:
            encoded_tags = str(encoded_tags)

        data.append([command_id, dataset, version, str(encoded_command), encoded_tags, str(world_before), source, str(location), reference, str(direction), command, str(tokens)])

    db.insert_many("ModelInput", data)


def prepare_data(version):
    #tokenizer = Tokenizer(engine = tokenization_engine, ignore_words = ignore_words, use_hunspell = use_hunspell or hunspell_first)
    #lemmatizer = LemmatizerTagger()
    #spell_checker = SpellChecker(vocabulary_version = version, tokenizer = tokenizer, lemmatizer = lemmatizer, occurences_to_repair = 30, use_hunspell = use_hunspell, use_synonyms = use_synonyms, use_lemma = use_lemma, max_levenshtein_distance = max_levenshtein, min_count = min_count, hunspell_first = hunspell_first)
    tokenizer, lemmatizer, spell_checker = get_tokenizer_lemmatizer_spellchecker(version)
    spell_checker.create_vocabulary()
    create_training_data(version = version, tokenizer = tokenizer, lemmatizer = lemmatizer, spell_checker = spell_checker)


def prepare_single_command(version, command):
    tokenizer, lemmatizer, spell_checker = get_tokenizer_lemmatizer_spellchecker(version)
    encoded_command, encoded_tags, tokens = encode_command(command, tokenizer, lemmatizer, spell_checker)
    return (encoded_command, encoded_tags)


def get_tokenizer_lemmatizer_spellchecker(version):
    setting = get_version_settings()[version]
    tokenizer = Tokenizer(engine = setting["tokenization"], ignore_words = setting["ignore_words"], use_hunspell = setting["use_hunspell"] or setting["hunspell_first"])
    lemmatizer = LemmatizerTagger()
    spell_checker = SpellChecker(vocabulary_version = version, tokenizer = tokenizer, lemmatizer = lemmatizer, occurences_to_repair = 30, use_hunspell = setting["use_hunspell"], use_synonyms = setting["use_synonyms"], use_lemma = setting["use_lemma"], max_levenshtein_distance = setting["max_levenshtein"], min_count = setting["min_count"], hunspell_first = setting["hunspell_first"])
    return tokenizer, lemmatizer, spell_checker


def version_3_ignore_words():
    return ["the", ".", "to", "so", ",", "s", "a", "e"]


def get_version_settings():
    version_settings = {}
    
    version_settings[10] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : False, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[11] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : False, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[12] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[13] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[14] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : True, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[15] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : True, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[16] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : False, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : True}
    version_settings[17] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : True, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[18] = {"tokenization" : "rule_based", "ignore_words" : version_3_ignore_words(), "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[19] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 1, "hunspell_first" : False}
    version_settings[20] = {"tokenization" : "rule_based", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 4, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[21] = {"tokenization" : "rule_based", "ignore_words" : version_3_ignore_words(), "use_hunspell" : True, "use_lemma" : True, "use_synonyms" : False, "min_count" : 4, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[22] = {"tokenization" : "rule_based", "ignore_words" : version_3_ignore_words(), "use_hunspell" : True, "use_lemma" : True, "use_synonyms" : True, "min_count" : 4, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[23] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : False, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : True}
    version_settings[24] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : True, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[25] = {"tokenization" : "udpipe", "ignore_words" : version_3_ignore_words(), "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[26] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 2, "max_levenshtein" : 1, "hunspell_first" : False}
    version_settings[27] = {"tokenization" : "udpipe", "ignore_words" : [], "use_hunspell" : True, "use_lemma" : False, "use_synonyms" : False, "min_count" : 4, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[28] = {"tokenization" : "udpipe", "ignore_words" : version_3_ignore_words(), "use_hunspell" : True, "use_lemma" : True, "use_synonyms" : False, "min_count" : 4, "max_levenshtein" : 0, "hunspell_first" : False}
    version_settings[29] = {"tokenization" : "udpipe", "ignore_words" : version_3_ignore_words(), "use_hunspell" : True, "use_lemma" : True, "use_synonyms" : True, "min_count" : 4, "max_levenshtein" : 0, "hunspell_first" : False}

    return version_settings


def main():
    #prepare_data(0, False, False, [])
    #prepare_data(1, True, False, [])
    #prepare_data(1, "rule_based", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2)
    #prepare_data(2, True, True, [])
    #prepare_data(3, True, True, version_3_ignore_words())
    #prepare_data(4, "rule_based", [], use_hunspell = True, use_lemma = True, use_synonyms = False, min_count = 2)
    #prepare_data(5, "rule_based", [], use_hunspell = True, use_lemma = True, use_synonyms = True, min_count = 5)
    #prepare_data(6, "rule_based", [], use_hunspell = False, use_lemma = False, use_synonyms = False, min_count = 2)
#    
#    prepare_data(10, "rule_based", [], use_hunspell = False, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(11, "udpipe", [], use_hunspell = False, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(12, "rule_based", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(13, "udpipe", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(14, "rule_based", [], use_hunspell = True, use_lemma = True, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(15, "udpipe", [], use_hunspell = True, use_lemma = True, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(16, "rule_based", [], use_hunspell = False, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = True)
#    prepare_data(17, "rule_based", [], use_hunspell = True, use_lemma = False, use_synonyms = True, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(18, "rule_based", version_3_ignore_words(), use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(19, "rule_based", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 1, hunspell_first = False)
#    prepare_data(20, "rule_based", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 4, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(21, "rule_based", version_3_ignore_words(), use_hunspell = True, use_lemma = True, use_synonyms = False, min_count = 4, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(22, "rule_based", version_3_ignore_words(), use_hunspell = True, use_lemma = True, use_synonyms = True, min_count = 4, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(23, "udpipe", [], use_hunspell = False, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = True)
#    prepare_data(24, "udpipe", [], use_hunspell = True, use_lemma = False, use_synonyms = True, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(25, "udpipe", version_3_ignore_words(), use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(26, "udpipe", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 2, max_levenshtein = 1, hunspell_first = False)
#    prepare_data(27, "udpipe", [], use_hunspell = True, use_lemma = False, use_synonyms = False, min_count = 4, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(28, "udpipe", version_3_ignore_words(), use_hunspell = True, use_lemma = True, use_synonyms = False, min_count = 4, max_levenshtein = 0, hunspell_first = False)
#    prepare_data(29, "udpipe", version_3_ignore_words(), use_hunspell = True, use_lemma = True, use_synonyms = True, min_count = 4, max_levenshtein = 0, hunspell_first = False)
#

    prepare_data(version = 22)   


if __name__ == "__main__":
    main()


