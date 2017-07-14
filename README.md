# Blockworld Commands

Code behind neural and other models for interpreting natural language commands using [Language Grounding dataset](https://nlg.isi.edu/language-grounding/).
Contains pretrained models which for source prediction achieves 98.8% accuracy and for location prediction achieves 68.9% accuracy (or 0.71 average distance between correct and predicted target location when using the distance metric).

## Getting started 

This guide was tested on clean installation of 64-bit Ubuntu 16.04.

### Prerequisities

You will need following:

* Linux OS
* Python 3
* Git
* Pip 3	
* Hunspell

To install the prerequisities (except Linux OS) on Ubuntu, use command:
`sudo apt-get install python3 git python3-pip libhunspell-dev`


### Installation
Get copy of the repository:

`git clone https://github.com/spekoun/blockworld_commands.git`

Optional: Download pretrained embeddings (822 MB). Standard models do not need them.
	
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

### Visual application
First script, *application.py*, launches the visualization of the world with possibility to try the models on existing or new commands.

### Working with models
The script *run_model.py* is used for training, analyzing and testing models.
It works with the database, which is by default saved in file *blockworld_commands/data/basic_database.db*.
By default this script trains our best location prediction models, which can take multiple days do finish.

To print all the options and their meaning use:
	
	`python3 run_model.py --help`

### Data preprocessing
There are only data preprocessed by a single version of the preprocessing process in the default database.
If you need the rest, use following scripts:
*load_data.py* loads data from the text files into the database.
*prepare_data.py* takes the loaded data, transforms them to be usable by the models (e.g. tokenization, spellchecking) and saves the results to the database.
This script needs a lot of time (approximately 1 hour).

## Author

Bedrich Pisl - bedapisl@gmail.com


