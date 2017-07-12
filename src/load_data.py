import json
import pdb
import gzip

from database import Database
from setting import digits, logos

data_dir = "../data"

datasets_filename = "set.json.gz"
sets = ["train", "test", "dev"]


def load_configuration_commands(json_data, configuration_id, db):
    data = []
    for command_group in json_data["notes"]:
        if command_group["type"] == "A0":
            for command_number, command in enumerate(command_group["notes"]):
                data.append([load_configuration_commands.command_id, configuration_id, command_number, command_group["start"], command_group["finish"], command.replace("\u200b", "").replace('\uf057', '')])
                load_configuration_commands.command_id += 1
    
    db.insert_many("Command", data)

load_configuration_commands.command_id = 0


def get_block_name(decoration, block_id):
    if decoration == "digit":
        return digits[block_id]

    elif decoration == "logo":
        return logos[block_id]

    else: 
        assert False


def load_configuration_world(json_data, configuration_id, db):

    data = []
    decoration = json_data["decoration"]
    side_length = json_data["side_length"]
    for state, state_data in enumerate(json_data["states"]):
        for block_id, [X, Y, Z] in enumerate(state_data):
            block_name = get_block_name(decoration, block_id)
            data.append([configuration_id, state, block_id, block_name, round(X / side_length, 4), round(Y / side_length, 4), round(Z / side_length, 4)])
    
    db.insert_many("World", data)


def main():
    db = Database()

    configuration_id = 0


    for dataset in sets:
        users = []
        data = []
        filename = data_dir + "/" + dataset + datasets_filename
        with gzip.open(filename, "rb") as input_file:
            for line in input_file.readlines():
                json_data = json.loads(line.decode("ascii"))

                for i in range(len(json_data["notes"])):
                    if json_data["notes"][i]["type"] == "A0":
                        users += json_data["notes"][i]["users"]

                
                db.insert("Configuration", [configuration_id, dataset, json_data["decoration"]])
                load_configuration_commands(json_data, configuration_id, db)
                load_configuration_world(json_data, configuration_id, db)
                configuration_id += 1
     
    db.commit()            
    

main()











