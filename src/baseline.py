

from database import Database
import pdb
from setting import digits, logos, directions#, compass_directions
import random


class DeterministicBaseline:
    def __init__(self, target):
        self.target = target
        

    def predict(self, commands, worlds, correct_source, correct_location, tags, logos, source_flags, dataset):#, raw_commands):
        correct_source = None
        correct_location = None
        dataset = None
        sources = []
        locations = []

        for k, (command, world, logo) in enumerate(zip(commands, worlds, logos)):
            sources.append(0)
            locations.append((0, 0))


        if self.target == "source":
            return sources, None
        elif self.target == "location":
            return None, locations
        
        assert False

    
    def get_reference(self, commands, world, source_id, location, tags, logos, source_flags, dataset):
        return [[]] * len(commands), [None] * len(commands), [None] * len(commands)


    def train(self, command, world, source_id, location, tags, logos, source_flags, learning_rate):
        pass
            

class RandomBaseline:
    def __init__(self, target):
        self.target = target
        

    def predict(self, commands, worlds, correct_source, correct_location, tags, logos, source_flags, dataset):#, raw_commands):
        correct_source = None
        correct_location = None
        dataset = None
        sources = []
        locations = []

        for k, (command, world, logo) in enumerate(zip(commands, worlds, logos)):
            sources.append(random.randint(0, 19))
            locations.append((random.uniform(-6.56, 6.56), random.uniform(-6.56, 6.56))) 


        if self.target == "source":
            return sources, None
        elif self.target == "location":
            return None, locations
        
        assert False

    
    def get_reference(self, commands, world, source_id, location, tags, logos, source_flags, dataset):
        return [[]] * len(commands), [None] * len(commands), [None] * len(commands)


    def train(self, command, world, source_id, location, tags, logos, source_flags, learning_rate):
        pass
 

