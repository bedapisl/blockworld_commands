from database import Database
import pdb
import re
from scipy.spatial import distance

from setting import digits, logos
from utils import levenshtein_distance


def tokenize(command):
    command = command.lower() + " "
    command = command.replace(",", " , ").replace("'s", "").replace("’s", "").replace("\"", "").replace("(", "").replace(")", "").replace(";", " ").replace(". ", " . ").replace("'", "")
    command = re.sub("(\D)-(\D)", "\g<1> \g<2>", command)       #substitutes - for space if there are not numbers before and after -

    tokens = command.split()
    return tokens


def create_vocabulary():
    db = Database()
    
    training_commands = db.get_all_rows_single_element("SELECT Command FROM Command JOIN Configuration ON Command.ConfigurationID = Configuration.ConfigurationID WHERE Configuration.Dataset = 'train'")

    tokens = {}
    for command in training_commands:
        for token in tokenize(command):
            if token in tokens:
                tokens[token] += 1
            else:
                tokens[token] = 1

    vocabulary_rows = []
    vocabulary_rows.append([0, "<bos>", len(training_commands)])
    vocabulary_rows.append([1, "<eos>", len(training_commands)])
    vocabulary_rows.append([2, "<unk>", len([token for (token, count) in tokens.items() if count == 1])])

    token_index = 3
    for token, count in tokens.items():
        if count > 1:
            vocabulary_rows.append([token_index, token, count])
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
def get_reference(command, decoration, location, source, world_after):
    possible_references = set(map(lambda token : get_possible_reference(token, decoration), tokenize(command)))

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


def encode_command(command, vocabulary):
    encoded = []
    encoded.append(vocabulary["<bos>"])

    for word in tokenize(command):
        if word in vocabulary:
            encoded.append(vocabulary[word])
        else:
            encoded.append(vocabulary["<unk>"])
    
    encoded.append(vocabulary["<eos>"])
    return encoded
    

def create_training_data():
    db = Database()
    all_commands = db.get_all_rows("SELECT Com.CommandID, Conf.Dataset, Conf.Decoration, Com.Command, Com.ConfigurationID, Com.Start, Com.Finish FROM Command AS Com JOIN Configuration AS Conf ON Com.ConfigurationID = Conf.ConfigurationID ORDER BY Com.CommandID")
    
    vocabulary = {}
    for (token_id, token) in db.get_all_rows("SELECT TokenID, Token FROM Vocabulary"):
        vocabulary[token] = token_id

    data = []
    for (command_id, dataset, decoration, command, configuration_id, start, finish) in all_commands:
        (world_before, world_after) = get_world(configuration_id, start, finish, db)
        source = get_changed_block(world_before, world_after)
        location = world_after[source]
        reference = get_reference(command, decoration, location, source, world_after)
        direction = get_direction(source, reference, world_after)
        encoded_command = encode_command(command, vocabulary)

        data.append([command_id, dataset, str(encoded_command), str(world_before), source, str(location), reference, str(direction), command])

    db.insert_many("ModelInput", data)



create_vocabulary()
create_training_data()


