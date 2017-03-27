import pdb
from setting import digits, logos

word_sets = {
"move" : [["move", "place", "put", "position"]],
"upper_lower" : [["upper", "top"], ["lower", "bottom"]],
"right_left" : [["right"], ["left"]],
"up_down" : [["up"], ["down"]],
"right_down_left_up" : [["right"], ["down"], ["left"], ["up"]],
"block" : [["block", "cube", ""]],
"the" : [["the", ""]],
"spaces" : [["spaces"], ["tiles"], ["blocks"], ["block lengths"], ["block spaces"]],
"so" : [["such that", "so", "so that"]],
"aligned" : [["aligned", "in line"]],
"direction" : [["right of", "east of"], ["below", "south of"], ["west of", "left of"], ["above", "north of"]]
}

block_name_digits = []
block_name_logos = []
numbers = [["zero", "0"]]

for i, digit in enumerate(digits):
    block_name_digits.append([])
    block_name_digits[i].append(digit)
    block_name_digits[i].append(str(i + 1))
    
    block_name_logos.append([])
    block_name_logos[i].append(logos[i])
    for logo_part in logos[i].split():
        block_name_logos[i].append(logo_part)

for i in range(0, 6):
    digit = digits[i]
    numbers.append([])
    numbers[-1].append(digit)
    numbers[-1].append(str(i + 1))


word_sets["block_name_digits"] = block_name_digits
word_sets["block_name_logos"] = block_name_logos
word_sets["number"] = numbers
 

def move_location(location, right_left, up_down, spaces = 1):
    spaces = spaces * 1.0936            #0.0936 is distance between blocks
    if right_left == 0:
        location[0] += spaces
    elif right_left == 1:
        location[0] -= spaces
    
    if up_down == 0:
        location[1] += spaces
    elif up_down == 1:
        location[1] -= spaces

    return location


def move_location_by_direction(location, direction, spaces = 1):
    if direction == 0:
        return move_location(location, 0, None, spaces)
    elif direction == 1:
        return move_location(location, None, 1, spaces)
    elif direction == 2:
        return move_location(location, 1, None, spaces)
    else:
        return move_location(location, None, 0, spaces)


def f0(world, words):
    location = world[words["block_name_2"]]
    location = move_location(location, words["right_left_2"], words["upper_lower_2"])
    return words["block_name_1"], location


def f1(world, words):
    location = move_location([0, 0], words["right_left_1"], words["upper_lower_1"], spaces = 6)
    return words["block_name_1"], location


def f2(world, words):
    location = world[words["block_name_1"]]
    location = move_location(location, None, words["up_down_1"], spaces = words["number_1"])
    location = move_location(location, words["right_left_1"], None, spaces = words["number_2"])
    return words["block_name_1"], location


def f3(world, words):
    location = [0, 0]
    location[0] = world[words["block_name_3"]][0]
    location[1] = world[words["block_name_2"]][1]
    return words["block_name_1"], location


def f4(world, words):
    location = world[words["block_name_2"]]
    location = move_location_by_direction(location, words["direction_1"], words["number_1"])
    return words["block_name_1"], location


def f5(world, words):
    location = world[words["block_name_1"]]
    location = move_location_by_direction(location, words["direction_1"])
    return words["block_name_2"], location


def f6(world, words):
    loc1 = world[words["block_name_1"]]
    loc2 = world[words["block_name_2"]]
    location = [0, 0]
    location[0] = (loc1[0] + loc2[0]) / 2
    location[1] = (loc1[1] + loc2[1]) / 2
    return words["block_name_3"], location


def f7(world, words):
    location = world[words["block_name_2"]]
    location = move_location_by_direction(location, words["direction_1"])
    location = move_location_by_direction(location, words["right_down_left_up_1"])
    return words["block_name_1"], location
    

world_transform_functions = [f0, f1, f2, f3, f4, f5, f6, f7]
