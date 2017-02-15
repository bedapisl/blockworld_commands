
import random
from database import Database
import ast
import numpy as np
import pdb
from setting import max_command_len, all_tags


class Dataset:
    def __init__(self, dataset, version, shuffle = True, dimension = 2, seed = 42):
        random.seed(seed)
        self.version = version
        self.instance_id = 0
        self.dataset_name = dataset

        db = Database()

        data = db.get_all_rows("""SELECT ModelInput.Command, WorldBefore, Source, Location, RawCommand, Decoration = 'logo', ModelInput.CommandID, Tags, Tokenized
                                    FROM ModelInput 
                                    JOIN Command ON Command.CommandID = ModelInput.CommandID
                                    JOIN Configuration ON Command.ConfigurationID = Configuration.ConfigurationID
                                    WHERE ModelInput.Dataset = '""" 
                                    + str(dataset) + "' AND Version = " + str(version) + " ORDER BY ModelInput.CommandID")

        if shuffle:
            random.shuffle(data)       

        self.commands = []
        self.worlds = []
        self.sources = []
        self.locations = []
        self.raw_commands = []
        self.logos = []
        self.command_ids = []
        self.tags = []
        self.tokenized = []

        for command, world, source_id, location, raw_command, logo, command_id, tag, tokens in data:
            command = ast.literal_eval(command)
            while len(command) < max_command_len:
                command.append(1)
            
            if len(command) > max_command_len:
                print("Error: Command length: " + str(len(command)) + " max: " + str(max_command_len))
                assert False

            self.commands.append(command)

            world = ast.literal_eval(world)
            new_world = []
            i = 0
            while len(new_world) < 20 * dimension:
                if i < len(world):
                    (x, y, z) = world[i]
                else:
                    (x, y, z) = (0, 0, 0)   #TODO solve the issue with missing blocks better

                if dimension == 2:
                    new_world.append(x)
                    new_world.append(z)

                elif dimension == 3:
                    new_world.append(x)
                    new_world.append(y)
                    new_world.append(z)

                else:
                    assert False

                i += 1

            self.worlds.append(new_world)
            self.sources.append(source_id)
            location = ast.literal_eval(location)
            if dimension == 2:
                self.locations.append([location[0], location[2]])
            else:
                self.locations.append(list(location))

            self.raw_commands.append(raw_command)
            self.logos.append(logo)
            self.command_ids.append(command_id)
            if tag is not None:
                tag = ast.literal_eval(tag)
            else:
                tag = []

            while len(tag) < max_command_len:
                tag.append(all_tags.index("X"))
            self.tags.append(tag)
            self.tokenized.append(tokens)
    

    def epoch_end(self):
        return self.instance_index == len(self.commands)

    
    def next_epoch(self):
        self.instance_index = 0
        self.shuffle()


    def get_next_batch(self, batch_size):
        batch_start = self.instance_index
        batch_end = min(batch_start + batch_size, len(self.commands))
        self.instance_index = batch_end

        return (np.array(self.commands[batch_start:batch_end]), np.array(self.worlds[batch_start:batch_end]), 
                    np.array(self.sources[batch_start:batch_end]), np.array(self.locations[batch_start:batch_end]), np.array(self.tags[batch_start:batch_end]))


    def get_all_data(self):
        return np.array(self.commands), np.array(self.worlds), np.array(self.sources), np.array(self.locations), np.array(self.tags)

    
    def get_raw_commands_and_logos(self):
        return self.raw_commands, self.logos, self.command_ids, self.tokenized

    
    def shuffle(self):
        index_shuf = list(range(len(self.commands)))
        random.shuffle(index_shuf)
        new_commands = []
        new_worlds = []
        new_sources = []
        new_locations = []
        new_raw_commands = []
        new_logos = []
        new_command_ids = []
        new_tags = []
        new_tokenized = []
        for i in index_shuf:
            new_commands.append(self.commands[i])
            new_worlds.append(self.worlds[i])
            new_sources.append(self.sources[i])
            new_locations.append(self.locations[i])
            new_raw_commands.append(self.raw_commands[i])
            new_logos.append(self.logos[i])
            new_command_ids.append(self.command_ids[i])
            new_tags.append(self.tags[i])
            new_tokenized.append(self.tokenized[i])
    
        self.commands = new_commands
        self.worlds = new_worlds
        self.sources = new_sources
        self.locations = new_locations
        self.raw_commands = new_raw_commands
        self.logos = new_logos
        self.command_ids = new_command_ids
        self.tags = new_tags
        self.tokenized = new_tokenized

