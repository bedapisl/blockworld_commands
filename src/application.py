import sys
from PyQt5.QtWidgets import QApplication, QWidget, QToolTip, QPushButton, QTextEdit, QLineEdit, QMessageBox
from PyQt5.QtGui import QFont, QImage, QPainter
from PyQt5.QtCore import pyqtRemoveInputHook, Qt
from dataset import Dataset
from database import Database 
from drawer import Drawer
from PIL import Image
from PIL.ImageQt import ImageQt
import pdb
import copy
import numpy as np
from utils import convert_world
from prepare_data import prepare_single_command
from run_model import load_args, load_model
from setting import max_command_len, all_tags


class ApplicationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.backend = ApplicationBackend()
        self.init_members()
        self.init_ui()
        self.load_command()


    def init_members(self):
        self.command_id = 1
        self.y_offset = 10
        self.x_offset = 10
        self.line_height = 30
        self.window_length = 1000
        self.window_height = 950
        self.default_command_id = 0
        self.world_before_showed = True
    
    
    def init_ui(self):
        QToolTip.setFont(QFont('SansSerif', 10))
       
        self.command_id_text = QLineEdit(self)
        self.command_id_text.resize(self.command_id_text.sizeHint())
        self.command_id_text.setFixedWidth(70)
        self.command_id_text.move(self.x_offset, self.y_offset)
        self.command_id_text.setText(str(self.default_command_id))

        self.load_command_button = QPushButton("Load Command", self)
        self.load_command_button.setToolTip("Loads command and world by ID")
        self.load_command_button.resize(self.load_command_button.sizeHint())
        self.load_command_button.move(self.x_offset + 90, self.y_offset)
        self.load_command_button.clicked.connect(self.load_command)

        self.process_button = QPushButton("Process", self)
        self.process_button.setToolTip("Change world based on current command")
        self.process_button.resize(self.process_button.sizeHint())
        self.process_button.move(self.x_offset, self.y_offset + 1 * self.line_height)
        self.process_button.clicked.connect(self.process)
 
        self.command_text = QLineEdit(self)
        self.command_text.resize(self.command_text.sizeHint())
        self.command_text.setFixedWidth(self.window_length - 2 * self.x_offset)
        self.command_text.move(self.x_offset, self.y_offset + 2 * self.line_height)

        self.back_button = QPushButton("<-", self)
        self.back_button.setToolTip("Show previous world state")
        self.back_button.resize(self.back_button.sizeHint())
        self.back_button.move(self.x_offset, self.y_offset + 3 * self.line_height)
        self.back_button.clicked.connect(self.back)
        #self.back_button.setEnabled(False)

        self.forward_button = QPushButton("->", self)
        self.forward_button.setToolTip("Show current world state")
        self.forward_button.resize(self.forward_button.sizeHint())
        self.forward_button.move(self.x_offset + 90, self.y_offset + 3 * self.line_height)
        self.forward_button.clicked.connect(self.forward)
        #self.forward_button.setEnabled(False)
        
        self.setGeometry(200, 100, self.window_length, self.window_height)
        self.setWindowTitle('Blockworld commands')
        self.show()

    
    def load_command(self):
        try:
            command_id = int(self.command_id_text.text())
            if command_id < 0 or command_id > self.backend.max_command_id:
                raise ValueError
            
            new_command = self.backend.load_command(command_id)
            self.command_text.setText(new_command)
            self.world_before_showed = True
            self.forward_button.setEnabled(False)
            self.back_button.setEnabled(False)
        
        except ValueError:
            QMessageBox.warning(self, "Warning", "Command ID must be value between 0 and " + str(self.backend.max_command_id))
    

    def forward(self):
        if self.backend.world_after is not None:
            self.world_before_showed = False
            self.back_button.setEnabled(True)
            self.forward_button.setEnabled(False)


    def back(self):
        self.world_before_showed = True
        self.back_button.setEnabled(False)
        self.forward_button.setEnabled(True)

    
    def process(self):
        if self.world_before_showed == False:
            self.backend.world_before = self.backend.world_after
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        current_command = self.command_text.text()
        self.backend.process(current_command)
        self.forward()
        QApplication.restoreOverrideCursor()


    def paintEvent(self, event):        #needed by Qt
        self.draw_image()


    def draw_image(self):
        image = self.backend.get_image(before = self.world_before_showed)
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(self.x_offset, self.y_offset + 4 * self.line_height, image)
        painter.end()
        self.show()
        self.update()


class ApplicationBackend:
    def __init__(self):
        self.source_model_id = 2001
        self.location_model_id = 2006
        self.max_command_id = 16766
        self.drawer = Drawer(output_dir = ".")
        self.source_preprocessing_version = self.get_preprocessing_version(self.source_model_id)
        self.location_preprocessing_version = self.get_preprocessing_version(self.location_model_id)
        self.world_before = None
        self.world_after = None
        self.source_model = load_model(load_args(self.source_model_id), self.source_model_id)
        self.location_model = load_model(load_args(self.location_model_id), self.location_model_id)
        
    
    def get_preprocessing_version(self, model_id):
        db = Database()
        return db.get_all_rows_single_element("SELECT Version FROM Results WHERE RunID = " + str(model_id))[0]
        
        
    def load_command(self, command_id):
        command_dataset = Dataset("single_command", self.source_preprocessing_version, specific_command = command_id)
        raw_commands, logos, _, _ = command_dataset.get_raw_commands_and_logos()
        self.logo = logos[0]
        _, worlds, sources, locations, _, _ = command_dataset.get_all_data()
        self.world_before = worlds[0]
        self.world_after = None
        return raw_commands[0]
        

    def get_image(self, before):
        if before:
            image = self.drawer.get_image(convert_world(self.world_before), self.logo)

        else:
            image = self.drawer.get_image(convert_world(self.world_after), self.logo)

        return QImage(ImageQt(image))

    
    def process(self, command):
        versions = [self.source_preprocessing_version, self.location_preprocessing_version]
        models = [self.source_model, self.location_model]
               
        for i in range(2):
            #pyqtRemoveInputHook()
            #pdb.set_trace()
            encoded_command, encoded_tags, _ = prepare_single_command(versions[i], command)
            while len(encoded_command) < max_command_len:
                encoded_command.append(1)
                encoded_tags.append(all_tags.index("X"))
           
            predicted_sources, predicted_locations = models[i].predict(np.array([encoded_command]), np.array([self.world_before]), np.array([-1]), np.array([[0, 0]]), np.array([encoded_tags]), np.array([self.logo]), "single_prediction", generate_summary = False)
            
            if i == 0:
                predicted_source = predicted_sources[0]
            else:
                predicted_location = predicted_locations[0][0]

        self.world_after = copy.deepcopy(self.world_before)
        self.world_after[2 * predicted_source] = predicted_location[0]
        self.world_after[2 * predicted_source + 1] = predicted_location[1]


if __name__ == '__main__':

    app = QApplication(sys.argv)
    gui = ApplicationGUI()

    sys.exit(app.exec_())






