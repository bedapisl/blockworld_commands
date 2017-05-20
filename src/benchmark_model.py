

from database import Database
import pdb
from setting import digits, logos, directions#, compass_directions


class BenchmarkModel:
    def __init__(self, version, target, dimension = 2):
        db = Database()
        
        self.target = target
        self.version = version
        self.dimension = dimension
        self.block_name_logos = []
        self.block_name_digits = []
        self.block_name_numerals = []
        self.block_name_digits_and_numerals = []
        self.direction_names = []
        self.no_block_words = [self.get_word_id("space", db), self.get_word_id("row", db), self.get_word_id("column", db)]

        for i, digit in enumerate(digits):
            self.block_name_digits_and_numerals.append([])
            self.block_name_digits_and_numerals[i].append(self.get_word_id(str(i + 1), db))
            self.block_name_digits_and_numerals[i].append(self.get_word_id(digit, db))

            self.block_name_digits.append([])
            self.block_name_digits[i].append(self.get_word_id(str(i + 1), db))
            
            self.block_name_numerals.append([])
            self.block_name_numerals[i].append(self.get_word_id(digit, db))

            self.block_name_logos.append([])
            self.block_name_logos[i].append(self.get_word_id(logos[i].replace(" ", ""), db))
            for logo_part in logos[i].split():
                self.block_name_logos[i].append(self.get_word_id(logo_part, db))
        
        for i in range(0, len(directions)):
            self.direction_names.append([])
            for direction in directions[i]:
                self.direction_names[i].append(self.get_word_id(direction, db))


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
            numerical_direction_names = self.block_name_digits_and_numerals
        else:
            block_names = self.block_name_digits_and_numerals
            numerical_direction_names = []
        
        blocks_in_command = []
        directions_in_command = []
        numerical_directions_in_command = []

        for i, word in enumerate(command):
            for block_id, block_name in enumerate(block_names):
                if i + 1 == len(command):
                    next_word = -1
                else:
                    next_word = command[i + 1]

                if word in block_name:
                    if next_word in self.no_block_words:
                        continue

                    if not logos and word in self.block_name_digits:        #pokud tam jsou cislice (napr. 4) tak budu ignorovat cislovky (napr. ctyri) jako bloky 
                        block_names = self.block_name_digits
                        numerical_direction_names = self.block_name_numerals
                    
                    if next_word in block_name and logos:
                        continue

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

            for numerical_direction_id, numerical_direction_name in enumerate(numerical_direction_names):
                if word in numerical_direction_name:
                    if get_index:
                        numerical_directions_in_command.append((i, numerical_direction_id))
                    else:
                        numerical_directions_in_command.append(numerical_direction_id)

        return (blocks_in_command, directions_in_command, numerical_directions_in_command)    


    def get_source_flags(self, source, command, logo):
        blocks_in_command, _, _ = self.get_blocks_and_directions(command, logo, get_index = True)

        source_flags = [0] * len(command)
        for block_order, block_id in blocks_in_command:
            if block_id == source:
                source_flags[block_order] = 1
     
        return source_flags
    

    
    def predict(self, commands, worlds, correct_source, correct_location, tags, logos, source_flags, dataset):#, raw_commands):
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

            blocks_in_command, directions_in_command, numerical_directions_in_command = self.get_blocks_and_directions(command, logo)

            if len(blocks_in_command) == 0:     #no block in command -> no change in world
                sources.append(0)               
                locations.append(world[0])
                continue

            source = blocks_in_command[0]
    
            blocks_in_command = [x for x in blocks_in_command if x != source]        #remove source

            if len(blocks_in_command) == 0:
                location = [0, 0]
            else:
                #location = world[blocks_in_command[-1]]
                location = self.get_center_of_gravity(world, blocks_in_command)

            if len(directions_in_command) == 0:     #no direction -> move directly to reference
                sources.append(source)
                locations.append(location)
                continue

            
            sources.append(source)
            direction = directions_in_command[-1]
            
            locations.append(location)
            block_distance = 1.0936

            use_numerical_direction = False
            if use_numerical_direction:
                if len(numerical_directions_in_command) > 0:
                    block_distance *= numerical_directions_in_command[-1]

            if direction == 0:
                locations[-1][0] -= block_distance

            if direction == 1:
                locations[-1][-1] += block_distance

            if direction == 2:
                locations[-1][0] += block_distance

            if direction == 3:
                locations[-1][-1] -= block_distance
            
            if direction == 4:
                locations[-1][0] -= block_distance
                locations[-1][-1] += block_distance
            
            if direction == 5:
                locations[-1][-1] += block_distance
                locations[-1][0] += block_distance

            if direction == 6:
                locations[-1][0] += block_distance
                locations[-1][-1] -= block_distance
 
            if direction == 7:
                locations[-1][-1] -= block_distance
                locations[-1][0] -= block_distance
           
        
        if self.target == "source":
            return sources, None
        elif self.target == "location":
            return None, locations
        
        assert False

    
    def get_reference(self, commands, world, source_id, location, tags, logos, source_flags, dataset):
        return [[]] * len(commands), [None] * len(commands), [None] * len(commands)


    def train(self, command, world, source_id, location, tags, logos, source_flags):
        pass
            


