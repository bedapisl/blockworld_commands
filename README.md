# Blockworld Commands

Code behind neural and other models for interpreting natural language commands using [Language Grounding dataset](https://nlg.isi.edu/language-grounding/).
Contains pretrained models which for the source prediction achieves 98.8% accuracy and for the location prediction achieves 68.9% accuracy (or 0.71 average distance between the correct and the predicted target location when using the distance metric).

Paper [**Communication with Robots using Multilayer Recurrent Networks**](http://www.aclweb.org/anthology/W17-2806), which was presented at workshop Language Grounding for Robotics at ACL 2017, used this source code for the experiments.

## Getting started 

This guide was tested on a clean installation of 64-bit Ubuntu 16.04.

### Prerequisities

You will need following:

* Linux OS
* Python 3
* Git
* Pip 3	
* Hunspell

To install the prerequisities (except Linux OS) on Ubuntu, use the command:

`sudo apt-get install python3 git python3-pip libhunspell-dev`

### Installation
Get copy of the repository:

`git clone https://github.com/bedapisl/blockworld_commands.git`

Optional: Download the pretrained embeddings (822 MB). The pretrained models do not need them.
	
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.100d.txt glove.6B.200d.txt glove.6B.300d.txt
mv glove.6B.50d.txt blockworld_commands/data
```

Go in the repository and install needed Python packages:
```
cd blockworld_commands
pip3 install -r requirements.txt
```

Test the application:
```
cd src
python3 application.py
```

## Usage

There are 4 scripts intended to be used by the user: *application.py*, *run_model.py*, *load_data.py* and *prepare_data.py*.

### Replicating results
This repository contains two pretrained models - one for source prediction and one for target prediction.
To load and evaluate the source model on test data use command:
```
python3 run_model.py --restore_and_test 5123
```

To load and evaluate the target model use:
```
python3 run_model.py --restore_and_test 5732
```

### Visual application
The first script, *application.py*, launches the visualization of the problem.
The application lets you load existing commands and world states.
Commands are identified with IDs, where train set commands have ID 0-11870, test set 11871-15047 and development set 15048-16766.

After pressing the *Process* button the best source and location model makes a prediction based on the current world state and sentence.
The predicted source block is then visualized in blue colour in the predicted location.

For better understanding the behaviour of the models, there is a textbox on the left showing predicted weights and a direction and a bottom textbox showing the preprocessed command.

### Working with models
The script *run_model.py* is used for training, analyzing and testing models.
It works with the database, which is by default saved in the file *blockworld_commands/data/basic_database.db*.
By default this script trains our best location prediction model, which can take multiple days do finish.

To print all the options and their meaning use:
	
`python3 run_model.py --help`

### Data preprocessing
There are only data preprocessed by a single version of the preprocessing process in the default database.
If you need the rest, use the following scripts:
*load_data.py* loads the data from the text files into the database.
*prepare_data.py* takes the loaded data, transforms them to be usable by the models (e.g. tokenization, spellchecking) and saves the results to the database.
This script needs a lot of time (approximately 1 hour).

### Troubleshooting
If you are unable to load models, make sure the correct version of Tensorflow  (1.0.1) is installed.

## Author

Bedrich Pisl - bedapisl@gmail.com
