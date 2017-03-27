

from database import Database
import pdb
from setting import digits, logos, directions, compass_directions


class BenchmarkModel:
    def __init__(self, version, target, dimension = 2):
        db = Database()
        
        self.target = target
        self.version = version
        self.dimension = dimension
        self.block_name_logos = []
        self.block_name_digits = []
        self.direction_names = []

        for i, digit in enumerate(digits):
            self.block_name_digits.append([])
            self.block_name_digits[i].append(self.get_word_id(digit, db))
            self.block_name_digits[i].append(self.get_word_id(str(i + 1), db))

            self.block_name_logos.append([])
            self.block_name_logos[i].append(self.get_word_id(logos[i].replace(" ", ""), db))
            for logo_part in logos[i].split():
                self.block_name_logos[i].append(self.get_word_id(logo_part, db))
        
        for i in range(0, len(directions)):
            self.direction_names.append([])
            self.direction_names[i].append(self.get_word_id(directions[i], db))
            self.direction_names[i].append(self.get_word_id(compass_directions[i], db))
            

    def get_word_id(self, word, db):
        word_id = db.get_all_rows_single_element("SELECT TokenID FROM Vocabulary WHERE Token = '" + str(word) + "' AND Version = " + str(self.version))
        if len(word_id) == 1:
            return word_id[0]

        return -1

    
    def predict(self, commands, worlds, correct_source, correct_location, tags, logos, dataset):#, raw_commands):
        correct_source = None
        correct_location = None
        dataset = None
        sources = []
        locations = []

        #number_of_blocks = [0] * 20
        
        #pdb.set_trace()

        for k, (command, world, logo) in enumerate(zip(commands, worlds, logos)):
        #    print(raw_commands[k])

            converted_world = []
            for i in range(0, len(world), self.dimension):
                converted_world.append([])
                for j in range(0, self.dimension):
                    converted_world[-1].append(world[i + j])

            world = converted_world

            if logo:
                block_names = self.block_name_logos
            else:
                block_names = self.block_name_digits

            blocks_in_command = []
            directions_in_command = []
            for word in command:
                for block_id, block_name in enumerate(block_names):
                    if word in block_name:
                        blocks_in_command.append(block_id)
                
                for direction_id, direction_names in enumerate(self.direction_names):
                    if word in direction_names:
                        directions_in_command.append(direction_id)
            
            #print(blocks_in_command)
            #pdb.set_trace()

            #number_of_blocks[len(set(blocks_in_command))] += 1
            #if len(set(blocks_in_command)) >= 3:
            #    pass
                #print(len(set(blocks_in_command)))
                #print(raw_commands[k])
 
            if len(blocks_in_command) == 0:     #no block in command -> no change in world
                sources.append(0)               
                locations.append(world[0])
                continue

            source = blocks_in_command[0]

            if len(blocks_in_command) == 1:
                location = [0, 0]
            else:
                location = world[blocks_in_command[-1]]

            if len(directions_in_command) == 0:     #no direction -> move directly to reference
                sources.append(source)
                locations.append(location)
                continue

            
            sources.append(source)
            direction = directions_in_command[-1]
            
            locations.append(location)
            block_distance = 1.0936
            if direction == 0:
                locations[-1][0] -= block_distance

            if direction == 1:
                locations[-1][-1] += block_distance

            if direction == 2:
                locations[-1][0] += block_distance

            if direction == 3:
                locations[-1][-1] -= block_distance

           
            #print(sources[-1])
            #print(command)
            #pdb.set_trace()
        
        #locations = [item for sublist in locations for item in sublist]     #flatten locations
        #print(number_of_blocks)
        if self.target == "source":
            return sources, None
        elif self.target == "location":
            return None, locations
        
        assert False


    def train(self, command, world, source_id, location, tags, logos):
        pass
            




            
        
        













