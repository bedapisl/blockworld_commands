import os
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import pdb

image_size = 1600

block_color = (255, 50, 50, 255)
highlight_color = (50, 50, 255, 255)
background_color = (200, 200, 200, 255)
block_size = (int(image_size / 40), int(image_size / 40))

class Drawer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logos_dir = "../data/logo_images/"
        self.logos = []
        self.highlight_logos = []
        self.font = PIL.ImageFont.truetype("../data/OpenSans-Regular.ttf", 20) 
        files = sorted(os.listdir(logos_dir))
        for image_file in files:
            full_image = Image.open(logos_dir + image_file, 'r')
            resized_image = full_image.resize(block_size, PIL.Image.ANTIALIAS)
            logo_image = Image.new('RGBA', block_size, block_color)
            logo_image.paste(resized_image, (0, 0), resized_image)
            self.logos.append(logo_image)
            highlight_logo_image = Image.new('RGBA', block_size, highlight_color)
            highlight_logo_image.paste(resized_image, (0, 0), resized_image)
            self.highlight_logos.append(highlight_logo_image)


        self.digits = []
        self.highlight_digits = []
        for i in range(1, 21):
            digit_image = Image.new('RGBA', block_size, block_color)
            draw = PIL.ImageDraw.Draw(digit_image)
            draw.text((2, 0), str(i), (0, 0, 0), font = self.font)
            self.digits.append(digit_image)

            highlight_digit_image = Image.new('RGBA', block_size, highlight_color)
            draw = PIL.ImageDraw.Draw(highlight_digit_image)
            draw.text((2, 0), str(i), (0, 0, 0), font = self.font)
            self.highlight_digits.append(highlight_digit_image)


    def draw_single_block(self, canvas, x, y, offset, highlight, logos, block_number):
        if logos:
            if highlight:
                image = self.highlight_logos[block_number]
            else:
                image = self.logos[block_number]
        else:
            if highlight:
                image = self.highlight_digits[block_number]
            else:
                image = self.digits[block_number]

        y = -y
      
        new_offset = []
        new_offset.append(int(offset[0] + x * block_size[0] + image_size / 4))
        new_offset.append(int(offset[1] + y * block_size[1] + image_size / 4))
        canvas.paste(image, tuple(new_offset))

        return canvas
    

    def draw_world(self, canvas, world, logos, offset, highlight = -1, predicted_source = -1, predicted_location = None):
        for i, (x, y) in enumerate(world):
            if i == highlight:
                canvas = self.draw_single_block(canvas, x, y, offset, True, logos, i)
            else:
                canvas = self.draw_single_block(canvas, x, y, offset, False, logos, i)

        if predicted_source != -1 and predicted_location is not None:
            (x, y) = predicted_location
            canvas = self.draw_single_block(canvas, x, y, offset, True, logos, predicted_source)

        return canvas


#            if logos:
#                if highlight == i:
#                    image = self.highlight_logos[i]
#                else:
#                    image = self.logos[i]
#            else:
#                if highlight == i:
#                    image = self.highlight_digits[i]
#                else:
#                    image = self.digits[i]
#
#            y = -y
#          
#            new_offset = []
#            new_offset.append(int(offset[0] + x * block_size[0] + image_size / 4))
#            new_offset.append(int(offset[1] + y * block_size[1] + image_size / 4))
#            canvas.paste(image, tuple(new_offset))
#        return canvas
    

    def get_image(self, world, logos, predicted_source = -1, predicted_location = None):
        canvas = Image.new('RGBA', (800, 800), background_color)
        draw = PIL.ImageDraw.Draw(canvas)
        canvas = self.draw_world(canvas, world, logos, [0, 0], highlight = -1, predicted_source = predicted_source, predicted_location = predicted_location)
        return canvas
 

    def save_image(self, command, world_before, world_after, logos, other_info, file_name, correct_location = None, moved_block = -1):
        canvas = Image.new('RGBA', (image_size, int(image_size / 2) + 200), background_color)

        draw = PIL.ImageDraw.Draw(canvas)
        draw.text((0, 0), command, (0, 0, 0), font = self.font)
        draw.text((0, 50), other_info, (0, 0, 0), font = self.font)

        canvas = self.draw_world(canvas, world_before, logos, [0, 200])
        canvas = self.draw_world(canvas, world_after, logos, [int(image_size / 2), 200], highlight = moved_block)

        if correct_location != None:
            offset = []
            offset.append(int(image_size / 2 + correct_location[0] * block_size[0] + image_size / 4))
            offset.append(int(200 - correct_location[1] * block_size[1] + image_size / 4))
            
            draw.rectangle((tuple(offset), (offset[0] + block_size[0], offset[1] + block_size[1])), fill = 'green')

        canvas.save(self.output_dir + "/" + file_name + ".png")


def test():
    drawer = Drawer("./images")
    command = "Test command blablabla"
    world_before = [(x, 0) for x in range(-10, 10)]
    world_after = [(0, y) for y in range(-10, 10)]

    drawer.save_image(command, world_before, world_after, True, "Other", "pokus")


if __name__ == "__main__":
    test()


