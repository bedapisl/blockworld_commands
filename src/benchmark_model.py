

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

    
    def get_center_of_gravity(self, world, blocks):
        x_coordinates = [world[i][0] for i in blocks]
        x = sum(x_coordinates) / len(blocks)
        y_coordinates = [world[i][1] for i in blocks]
        y = sum(y_coordinates) / len(blocks)
        return [x, y]


    def get_blocks_and_directions(self, command, logos, get_index = False):
        if logos:
            block_names = self.block_name_logos
        else:
            block_names = self.block_name_digits

        blocks_in_command = []
        directions_in_command = []
        for i, word in enumerate(command):
            for block_id, block_name in enumerate(block_names):
                if word in block_name:
                    if get_index:
                        blocks_in_command.append((i, block_id))
                    else:
                        blocks_in_command.append(block_id)
            

            for direction_id, direction_names in enumerate(self.direction_names):
                if word in direction_names:
                    if get_index:
                        directions_in_command.append((i, direction_id))
                    else:
                        directions_in_command.append(direction_id)
               
        return (blocks_in_command, directions_in_command)    

    
    def predict(self, commands, worlds, correct_source, correct_location, tags, logos, dataset):#, raw_commands):
        correct_source = None
        correct_location = None
        dataset = None
        sources = []
        locations = []

        for k, (command, world, logo) in enumerate(zip(commands, worlds, logos)):

            converted_world = []
            for i in range(0, len(world), self.dimension):
                converted_world.append([])
                for j in range(0, self.dimension):
                    converted_world[-1].append(world[i + j])

            world = converted_world

            blocks_in_command, directions_in_command = self.get_blocks_and_directions(command, logo)
#
#            if logo:
#                block_names = self.block_name_logos
#            else:
#                block_names = self.block_name_digits
#
#            blocks_in_command = []
#            directions_in_command = []
#            for word in command:
#                for block_id, block_name in enumerate(block_names):
#                    if word in block_name:
#                        blocks_in_command.append(block_id)
#                
#                for direction_id, direction_names in enumerate(self.direction_names):
#                    if word in direction_names:
#                        directions_in_command.append(direction_id)
#            
            if len(blocks_in_command) == 0:     #no block in command -> no change in world
                sources.append(0)               
                locations.append(world[0])
                continue

            source = blocks_in_command[0]

            if len(blocks_in_command) == 1:
                location = [0, 0]
            else:
                location = world[blocks_in_command[-1]]
                #location = self.get_center_of_gravity(world, blocks_in_command[1:])

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
           
        
        if self.target == "source":
            return sources, None
        elif self.target == "location":
            return None, locations
        
        assert False


    def train(self, command, world, source_id, location, tags, logos):
        pass
            


