import pdb
import random
import copy
from config_generator import word_sets, world_transform_functions
from drawer import Drawer


class SentenceSkeleton:
    def __init__(self, words):
        self.words = []
        for word in words:
            if word[0] == "[":
                word = word[1:-1]
                self.words.append((word, True))
            else:
                self.words.append((word, False))


    def new_sentence(self, logos):
        sentence = ""
        word_dict = {}
        chosen_words = {}
        already_used_meanings = {}
        for (word, use_set) in self.words:
            if use_set == False:
                chosen_word = word
            else:
                if word in chosen_words:
                    chosen_word = chosen_words[word]
                else:
                    chosen_word, meaning = self.random_word(word, logos, already_used_meanings)
                    chosen_words[word] = chosen_word
                    word_dict[word] = meaning
               
            sentence += chosen_word
            if chosen_word != "":
                sentence += " "
        
        return sentence[0:-1], word_dict

    
    def random_word(self, word, logos, already_used_meanings):
        if word[-1].isdigit():
            word = word[0:-2]
   
        if word == "block_name":
            if logos:
                word += "_logos"
            else:
                word += "_digits"

        word_set = word_sets[word]
        chosen_word_meaning = random.randint(0, len(word_set) - 1)

        if word in already_used_meanings.keys():                             #different tokens (e.g. [block_name_1] and [block_name_2]) must have different representation (e.g. adidas and bmw)
            while chosen_word_meaning in already_used_meanings[word]:
                chosen_word_meaning = random.randint(0, len(word_set) - 1)
            already_used_meanings[word].append(chosen_word_meaning)
        else:
            already_used_meanings[word] = [chosen_word_meaning]

        synonyms = word_set[chosen_word_meaning]
        chosen_word = synonyms[random.randint(0, len(synonyms) - 1)]
        
        return chosen_word, chosen_word_meaning
        
        
class Generator:
    def __init__(self, seed):
        random.seed(seed)
        self.lower_bound = -6.5     #world size
        self.upper_bound = 6.5      #world size
        self.seed = seed
        self.skeletons = self.load_skeletons()

    
    def remove_empty_strings(self, array):
        return [element for element in array if len(element) > 0]

    
    def load_skeletons(self):
        skeletons = []
        self.skeleton_weight_sum = 0
        
        input_file = open("../data/sentence_skeletons", "r")
        lines = input_file.readlines()
        
        i = 0
        while i < len(lines):
            words = self.remove_empty_strings(lines[i].split())
            if len(words) == 0:
                i += 1
                continue

            skeleton = SentenceSkeleton(words)
            words = self.remove_empty_strings(lines[i + 1].split())
            function_index = int(words[0])
            skeleton_weight = float(words[1])
            skeletons.append((skeleton, world_transform_functions[function_index], skeleton_weight))
            self.skeleton_weight_sum += skeleton_weight

            i += 2

        return skeletons
    

    def detect_collision(self, world, new_coordinates):
        for i in range(0, len(world)):
            if abs(world[i][0] - new_coordinates[0]) <= 1 and abs(world[i][1] - new_coordinates[1]) <= 1:
                return True

        return False

    def random_world(self):
        world = []
        while len(world) < 20:
            new_coordinates = [random.uniform(self.lower_bound, self.upper_bound), random.uniform(self.lower_bound, self.upper_bound)]
           
            if not self.detect_collision(world, new_coordinates):
                world.append(new_coordinates)
                
        return world


    def new_sentence(self):
        random_selection = random.uniform(0, self.skeleton_weight_sum)
        weight_sum = 0
        skeleton = None
        transform_function = None
        for skeleton_element, transform_function_element, skeleton_weight in self.skeletons:
            weight_sum += skeleton_weight
            if weight_sum > random_selection:
                skeleton = skeleton_element
                transform_function = transform_function_element
                break
            
        world_before = self.random_world()
        logos = bool(random.randint(0, 1))
        sentence, word_dict = skeleton.new_sentence(logos)
        source, location = transform_function(copy.deepcopy(world_before), word_dict)

        if source < 0 or source > 20 or location[0] < self.lower_bound or location[0] > self.upper_bound or location[1] < self.lower_bound or location[1] > self.upper_bound:
            return self.new_sentence()
        
        if self.detect_collision(world_before, location):
            return self.new_sentence()

        return sentence, world_before, source, location, logos


if __name__ == "__main__":
    generator = Generator(42)
    drawer = Drawer("./images/generated/")

    for i in range(0, 20):
        sentence, world_before, source, location, logos = generator.new_sentence()
        world_after = copy.deepcopy(world_before)
        world_after[source] = location
        drawer.save_image(sentence, world_before, world_after, logos, "", str(i), moved_block = source)





