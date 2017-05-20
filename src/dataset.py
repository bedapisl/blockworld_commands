
import random
from database import Database
import ast
import numpy as np
import pdb
from setting import max_command_len, all_tags
import copy
from benchmark_model import BenchmarkModel


class Dataset:
    def __init__(self, dataset, version, shuffle = True, dimension = 2, seed = 42, specific_command = None, generated_commands = 0, switch_blocks = 0.0):
        random.seed(seed)
        self.version = version
        self.dataset_name = dataset
        self.switch_blocks = switch_blocks
        self.benchmark = BenchmarkModel(version, "source")

        db = Database()

        if specific_command is None:
            data = db.get_all_rows("""SELECT ModelInput.Command, WorldBefore, Source, PredictedSource, Location, RawCommand, Decoration = 'logo', ModelInput.CommandID, Tags, Tokenized
                                        FROM ModelInput 
                                        JOIN Command ON Command.CommandID = ModelInput.CommandID
                                        JOIN Configuration ON Command.ConfigurationID = Configuration.ConfigurationID
                                        WHERE ModelInput.Dataset = '""" 
                                        + str(dataset) + "' AND Version = " + str(version) + " ORDER BY ModelInput.CommandID")
        else:
            data = db.get_all_rows("""SELECT ModelInput.Command, WorldBefore, Source, PredictedSource, Location, RawCommand, Decoration = 'logo', ModelInput.CommandID, Tags, Tokenized
                                        FROM ModelInput 
                                        JOIN Command ON Command.CommandID = ModelInput.CommandID
                                        JOIN Configuration ON Command.ConfigurationID = Configuration.ConfigurationID
                                        WHERE ModelInput.CommandID = '""" 
                                        + str(specific_command) + "' AND Version = " + str(version))


        if generated_commands > 0:
            from generator import Generator
            from prepare_data import SingleCommandEncoder
           
            generator = Generator(seed = seed)
            single_command_encoder = SingleCommandEncoder()
            
            for i in range(generated_commands):
                sentence, world_before, source, location, logos = generator.new_sentence()
                encoded_command, tags, tokens = single_command_encoder.prepare_single_command(version, sentence)
                data.append((str(encoded_command), str(world_before), source, str(location), sentence, logos, -1, str(tags), str(tokens)))
        
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
        self.source_flags = []

        for command, world, source_id, predicted_source_id, location, raw_command, logo, command_id, tag, tokens in data:
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
            self.source_flags.append(self.benchmark.get_source_flags(predicted_source_id, command, logo))


    def epoch_end(self):
        return self.instance_index == len(self.commands)

    
    def next_epoch(self):
        self.instance_index = 0
        self.shuffle()


    def get_next_batch(self, batch_size):
        batch_start = self.instance_index
        batch_end = min(batch_start + batch_size, len(self.commands))
        self.instance_index = batch_end

        results_order = [self.commands, self.worlds, self.sources, self.locations, self.tags, self.logos, self.source_flags]
        results = []

        for single_result in results_order:
            results.append(np.array(single_result[batch_start:batch_end]))

        if self.switch_blocks > 0.0:
            switched_commands = []
            switched_worlds = []
            for i in range(batch_start, batch_end):
                if self.switch_blocks > random.uniform(0,1):
                    #print(self.worlds[i])
                    #print(self.raw_commands[i])
                    #print(self.commands[i])
                    switched_world, switched_command = self.get_switched_blocks(i)
                    #print(switched_world)
                    #print(switched_command)
                    #pdb.set_trace()
                    switched_commands.append(switched_command)
                    switched_worlds.append(switched_world)
                else:
                    switched_commands.append(self.commands[i])
                    switched_worlds.append(self.worlds[i])

            results[0] = np.array(switched_commands)
            results[1] = np.array(switched_worlds)

        return tuple(results)

        #return (np.array(self.commands[batch_start:batch_end]), np.array(self.worlds[batch_start:batch_end]), 
        #            np.array(self.sources[batch_start:batch_end]), np.array(self.locations[batch_start:batch_end]), 
        #            np.array(self.tags[batch_start:batch_end]), np.array(self.logos[batch_start:batch_end]))


    def get_all_data(self):
        return np.array(self.commands), np.array(self.worlds), np.array(self.sources), np.array(self.locations), np.array(self.tags), np.array(self.logos), np.array(self.source_flags)

    
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
        new_source_flags = []
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
            new_source_flags.append(self.source_flags[i])
    
        self.commands = new_commands
        self.worlds = new_worlds
        self.sources = new_sources
        self.locations = new_locations
        self.raw_commands = new_raw_commands
        self.logos = new_logos
        self.command_ids = new_command_ids
        self.tags = new_tags
        self.tokenized = new_tokenized
        self.source_flags = new_source_flags

    
    def get_switched_blocks(self, command_index):
        old_world = self.worlds[command_index]
        old_command = self.commands[command_index]
        new_world = copy.deepcopy(old_world)
        new_command = copy.deepcopy(old_command)

        index_shuf = list(range(20))
        random.shuffle(index_shuf)
    
        blocks_in_command, _, _ = self.benchmark.get_blocks_and_directions(old_command, self.logos[command_index], get_index = True)

        if self.logos[command_index]:
            block_names = self.benchmark.block_name_logos
        else:
            block_names = self.benchmark.block_name_digits

        for i, j in enumerate(index_shuf):
            new_world[2 * j] = old_world[2 * i]
            new_world[2 * j + 1] = old_world[2 * i + 1]

            for word_index, block_index in blocks_in_command:
                if block_index == i:
                    names_for_block = block_names[j]
                    new_command[word_index] = names_for_block[random.randint(0, len(names_for_block) - 1)]

        return new_world, new_command

            
            


