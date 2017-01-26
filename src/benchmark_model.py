

from database import Database
import pdb
from setting import digits, logos, directions, compass_directions


class BenchmarkModel:
    def __init__(self, version, dimension = 2):
        db = Database()
        
        self.version = version
        self.dimension = dimension
        self.block_names = []
        self.direction_names = []

        for i, digit in enumerate(digits):
            self.block_names.append([])
            self.block_names[i].append(self.get_word_id(digit, db))
            self.block_names[i].append(self.get_word_id(str(i + 1), db))
            for logo_part in logos[i].split():
                self.block_names[i].append(self.get_word_id(logo_part, db))
        
        for i in range(0, len(directions)):
            self.direction_names.append([])
            self.direction_names[i].append(self.get_word_id(directions[i], db))
            self.direction_names[i].append(self.get_word_id(compass_directions[i], db))
            

    def get_word_id(self, word, db):
        word_id = db.get_all_rows_single_element("SELECT TokenID FROM Vocabulary WHERE Token = '" + str(word) + "' AND Version = " + str(self.version))
        if len(word_id) == 1:
            return word_id[0]

        return -1

    
    def predict(self, commands, worlds, correct_source, correct_location, dataset):
        correct_source = None
        correct_location = None
        dataset = None
        sources = []
        locations = []

        for command, world in zip(commands, worlds):
            converted_world = []
            for i in range(0, len(world), self.dimension):
                converted_world.append([])
                for j in range(0, self.dimension):
                    converted_world[-1].append(world[i + j])

            world = converted_world

            blocks_in_command = []
            directions_in_command = []
            for word in command:
                for block_id, block_names in enumerate(self.block_names):
                    if word in block_names:
                        blocks_in_command.append(block_id)
                
                for direction_id, direction_names in enumerate(self.direction_names):
                    if word in direction_names:
                        directions_in_command.append(direction_id)
            
            if len(blocks_in_command) == 0:     #no block in command -> no change in world
                sources.append(0)               
                locations.append(world[0])
                continue

            source = blocks_in_command[0]
            reference = blocks_in_command[-1]

            if len(directions_in_command) == 0:     #no direction -> move directly to reference
                sources.append(source)
                locations.append(world[reference])
                continue

            
            sources.append(source)
            direction = directions_in_command[-1]
            
            locations.append(world[reference])
            if direction == 0:
                locations[-1][0] -= 1

            if direction == 1:
                locations[-1][-1] += 1

            if direction == 2:
                locations[-1][0] += 1

            if direction == 3:
                locations[-1][-1] -= 1

            #print(sources[-1])
            #print(command)
            #pdb.set_trace()
        
        #locations = [item for sublist in locations for item in sublist]     #flatten locations
        return sources, locations


    def train(self, command, world, source_id, location):
        pass
            




            
        
        













