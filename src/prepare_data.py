from database import Database
import pdb
import re
import hunspell
from scipy.spatial import distance

from setting import digits, logos, all_tags
from utils import levenshtein_distance

from ufal.udpipe import *


class Tokenizer():
    def __init__(self, spell_check, use_udpipe, ignore_words):
        self.spell_check = spell_check
        self.use_udpipe = use_udpipe
        self.ignore_words = ignore_words
        self.changes = {}
        if self.spell_check:
            self.checker = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')
            self.checker.add('artois')
            self.checker.add(',')
            self.checker.add('starbucks')
            self.checker.add('esso')
            self.checker.add('nvidia')
            self.checker.add('sri')
            self.checker.add('bmw')
            self.checker.add('heineken')
            self.checker.add('mercedes')

        if self.use_udpipe:
            self.model = Model.load("../data/english-ud-1.2-160523.udpipe")
            self.pipeline = Pipeline(self.model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")

    
    def correct(self, token):
        try:
            if not self.checker.spell(token):
                suggestions = self.checker.suggest(token)
                if len(suggestions) > 0:
                    self.changes[token] = suggestions[0].decode()
                    return suggestions[0].decode().lower()
        
        except UnicodeEncodeError:  
            pass

        return token

    
    def udpipe_process(self, command):
        processed = self.pipeline.process(command.lower())
        words = processed.split("\n")
        base_forms = []
        tags = []

        for word in words:
            parts = word.split("\t")
            if len(parts) < 3:
                continue
            
            if parts[1].lower() in ["ups", "sri"]:              #udpipe uncorrectly changes these to "up" and "sr"
                base_forms.append(parts[1].lower())
            else:
                base_forms.append(parts[2].lower())
            tags.append(parts[3])

        return base_forms, tags


    def tokenize(self, command):
        use_tags = False
        if self.use_udpipe:
            tokens, tags = self.udpipe_process(command)

            new_tokens = []
            new_tags = []
            for i, token in enumerate(tokens):
                token = re.sub("([0-9])([a-zA-Z])", "\g<1> \g<2>", token)
                token_parts = token.replace("-", " ").replace("'", " ").replace("’", " ").replace("hp.", "hp").split(" ")
                for token_part in token_parts:
                    if token_part != '':
                        new_tokens.append(token_part)
                        new_tags.append(tags[i])

            tokens = new_tokens
            tags = new_tags
            use_tags = True

        else:
            command = command.lower() + " "
            command = command.replace(",", " , ").replace("'s", "").replace("’s", "").replace("\"", "").replace("(", "").replace(")", "").replace(";", " ").replace(". ", " . ").replace("'", "")
            command = re.sub("(\D)-(\D)", "\g<1> \g<2>", command)       #substitutes - for space if there are not numbers before and after -
            command = re.sub("([a-zA-Z])([0-9])", "\g<1> \g<2>", command)
            command = re.sub("([0-9])([a-zA-Z])", "\g<1> \g<2>", command)

            tokens = command.split()
            tags = None
        
        if self.spell_check:
            corrected_tokens = []
            new_tags = []
            for i, token in enumerate(tokens):
                for token_part in self.correct(token).split(" "):
                    if token_part not in self.ignore_words:
                        corrected_tokens.append(token_part)
                        if tags is not None:
                            new_tags.append(tags[i])
            
            tokens = corrected_tokens
            tags = new_tags

        if use_tags:
            assert len(tags) == len(tokens)
        else:
            tags = None

        return tokens, tags


def create_vocabulary(version, tokenizer):
    db = Database()
    
    training_commands = db.get_all_rows_single_element("SELECT Command FROM Command JOIN Configuration ON Command.ConfigurationID = Configuration.ConfigurationID WHERE Configuration.Dataset = 'train'")

    tokens = {}
    for command in training_commands:
        command_tokens, _ = tokenizer.tokenize(command)
        for token in command_tokens:
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1

    #print(tokenizer.changes)

    vocabulary_rows = []
    vocabulary_rows.append([0, "<bos>", len(training_commands), version])
    vocabulary_rows.append([1, "<eos>", len(training_commands), version])
    vocabulary_rows.append([2, "<unk>", len([token for (token, count) in tokens.items() if count == 1]), version])

    token_index = 3
    for token, count in tokens.items():
        if count > 1:
            vocabulary_rows.append([token_index, token, count, version])
            token_index += 1

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


def encode_command(command, vocabulary, tokenizer):
    encoded = []
    encoded.append(vocabulary["<bos>"])

    tokens, command_tags = tokenizer.tokenize(command)

    for i, word in enumerate(tokens):
        if word in vocabulary:
            encoded.append(vocabulary[word])
        else:
            encoded.append(vocabulary["<unk>"])
            tokens[i] = "<unk>(" + word + ")"

    
    encoded.append(vocabulary["<eos>"])

    if command_tags is not None:
        command_tags.insert(0, "X")
        command_tags.append("X")
        
        encoded_tags = []
        for tag in command_tags:
            encoded_tags.append(all_tags.index(tag))

    else:
        encoded_tags = None

    return encoded, encoded_tags, tokens
    

def create_training_data(version, tokenizer):
    db = Database()
    all_commands = db.get_all_rows("SELECT Com.CommandID, Conf.Dataset, Conf.Decoration, Com.Command, Com.ConfigurationID, Com.Start, Com.Finish FROM Command AS Com JOIN Configuration AS Conf ON Com.ConfigurationID = Conf.ConfigurationID ORDER BY Com.CommandID")
    
    vocabulary = {}
    for (token_id, token) in db.get_all_rows("SELECT TokenID, Token FROM Vocabulary WHERE Version = " + str(version)):
        vocabulary[token] = token_id

    data = []
    for (command_id, dataset, decoration, command, configuration_id, start, finish) in all_commands:
        (world_before, world_after) = get_world(configuration_id, start, finish, db)
        source = get_changed_block(world_before, world_after)
        location = world_after[source]
        reference = get_reference(command, decoration, location, source, world_after, tokenizer)
        direction = get_direction(source, reference, world_after)
        encoded_command, encoded_tags, tokens = encode_command(command, vocabulary, tokenizer)

        if encoded_tags is not None:
            encoded_tags = str(encoded_tags)

        data.append([command_id, dataset, version, str(encoded_command), encoded_tags, str(world_before), source, str(location), reference, str(direction), command, str(tokens)])

    db.insert_many("ModelInput", data)


def prepare_data(version, spell_check, use_udpipe, ignore_words):
    tokenizer = Tokenizer(spell_check = spell_check, use_udpipe = use_udpipe, ignore_words = ignore_words)
    create_vocabulary(version = version, tokenizer = tokenizer)
    create_training_data(version = version, tokenizer = tokenizer)


def version_3_ignore_words():
    return ["the", ".", "to", "so", ",", "s", "a", "e"]


def main():
    #prepare_data(0, False, False, [])
    #prepare_data(1, True, False, [])
    #prepare_data(2, True, True, [])
    #prepare_data(3, True, True, version_3_ignore_words())

main()


