
import tensorflow as tf
from setting import max_command_len
import numpy as np
import pdb
from utils import get_embedding_matrix, get_word_characters, vocabulary_length

class Network:
    def __init__(self, network_type, run_id, learning_rate, target, rnn_cell_type, rnn_cell_dim, bidirectional, hidden_dimension, 
                            use_world, dropout_input, dropout_output, embeddings, version, threads, use_tags, rnn_output, hidden_layers = 2, world_dimension = 2):
        self.target = target
        self.dropout_input = dropout_input
        self.dropout_output = dropout_output
        self.use_tags = use_tags
        self.rnn_layers = 0

        graph = tf.Graph()
        graph.seed = 42
        self.session = tf.Session(graph = graph, config = tf.ConfigProto(inter_op_parallelism_threads = threads, intra_op_parallelism_threads = threads))

        self.summary_writer = tf.summary.FileWriter("logs/" + str(run_id) + "_" + target + "_" + network_type, flush_secs=10)

        with self.session.graph.as_default():
            if rnn_cell_type == "LSTM":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)

            elif rnn_cell_type == "GRU":
                rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)

            else:
                raise ValueError("RNN cell type must be 'LSTM' or 'GRU'")

            self.command = tf.placeholder(tf.int32, [None, max_command_len])          #[batch, sentence_length]
            self.command_lens = tf.placeholder(tf.int32, [None])
            self.tags = tf.placeholder(tf.int32, [None, max_command_len])
            self.world = tf.placeholder(tf.float32, [None, 20 * world_dimension])     #[batch, block_coordinates]
            self.source = tf.placeholder(tf.int32, [None])
            self.location = tf.placeholder(tf.float32, [None, world_dimension])

            self.dropout_input_tensor = tf.placeholder(tf.float32, shape = ())
            self.dropout_output_tensor = tf.placeholder(tf.float32, shape = ())
            self.dropout_input_multiplier = tf.placeholder(tf.float32, shape = ())
            self.dropout_output_multiplier = tf.placeholder(tf.float32, shape = ())

            ################################## EMBEDDINGS ############################
            embedding_dim = 50
            if embeddings == "none":
                self.embedded_words = tf.one_hot(self.command, vocabulary_length(version))

            elif embeddings == "random":
                self.embeddings = tf.Variable(tf.random_uniform([vocabulary_length(version), embedding_dim], minval = -1, maxval = 1))
                self.embedded_words = tf.gather(self.embeddings, self.command)

            elif embeddings == "pretrained":
                self.embeddings = tf.Variable(initial_value = get_embedding_matrix(version), dtype = tf.float32)
                self.embedded_words = tf.gather(self.embeddings, self.command)

            elif embeddings == "character":
                char_rnn_cell = tf.nn.rnn_cell.GRUCell(embedding_dim)
                word_to_chars, num_chars, word_lens = get_word_characters(version)
                self.word_to_chars = tf.constant(word_to_chars)     # [number_of_words, max_word_len]
                self.word_lens = tf.constant(word_lens)             # [number_of_words]
                self.command_in_chars = tf.gather(self.word_to_chars, self.command) # [batch_size, max_command_len, max_word_len]
                self.command_word_lens = tf.gather(self.word_lens, self.command)    # [batch_size, max_command_len]

                self.character_embeddings = tf.Variable(tf.random_uniform([num_chars, embedding_dim], minval = -1, maxval = 1))
                self.embedded_chars = tf.gather(self.character_embeddings, self.command_in_chars)       # [batch_size, max_command_len, max_word_len, embedding_dim]
                self.embedded_chars_concat = tf.reshape(self.embedded_chars, [-1, max(word_lens), embedding_dim])
                self.command_word_lens = tf.reshape(self.command_word_lens, [-1])
                _, self.embedded_words = tf.nn.dynamic_rnn(char_rnn_cell, self.embedded_chars_concat, sequence_length = self.command_word_lens, dtype = tf.float32)
                self.embedded_words = tf.reshape(self.embedded_words, [-1, max_command_len, embedding_dim])
            
            ################################# TAGS ####################################
            if use_tags:
                tag_dim = 10
                self.tag_embeddings = tf.Variable(tf.random_uniform([vocabulary_length(version), tag_dim], minval = -1, maxval = 1))
                self.embedded_tags = tf.gather(self.tag_embeddings, self.tags)
                self.embedded_words = tf.concat(2, [self.embedded_words, self.embedded_tags])
            
            ############################### DROPOUT INPUT ############################
            if target == "location":
                self.one_hot_words = tf.scalar_mul(self.dropout_input_multiplier, tf.nn.dropout(self.embedded_words, 1.0 - self.dropout_input_tensor))
            else:
                self.one_hot_words = tf.nn.dropout(self.embedded_words, 1.0 - self.dropout_input_tensor)
            

            ############################## HIDDEN LAYER #############################
            if network_type == "rnn":
               
                if bidirectional:
                    self.rnn_input = self.one_hot_words
                    #self.bidirectional_rnn_output, self.bidirectional_rnn_state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, self.one_hot_words, sequence_length = self.command_lens, dtype = tf.float32, scope = "hidden_layer_0")

                    for i in range(0, hidden_layers):
                        #self.previous_state = tf.concat(1, [self.bidirectional_rnn_output[0], self.bidirectional_rnn_output[1]])
                        self.bidirectional_rnn_output, self.bidirectional_rnn_state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, self.rnn_input, sequence_length = self.command_lens, dtype = tf.float32, scope = "hidden_layer_" + str(i))
                        self.rnn_input = tf.concat(2, [self.bidirectional_rnn_output[0], self.bidirectional_rnn_output[1]])

                    if rnn_output == "last_state":
                        if rnn_cell_type == "LSTM":
                            self.rnn_output = tf.concat(1, [self.bidirectional_rnn_state[0].c, self.bidirectional_rnn_state[1].c])
                        else:
                            self.rnn_output = tf.concat(1, self.bidirectional_rnn_state)
                    elif rnn_output == "all_outputs":
                        self.concatenated_outputs = tf.concat(1, [self.bidirectional_rnn_output[0], self.bidirectional_rnn_output[1]])
                        self.rnn_output = tf.reduce_sum(self.concatenated_outputs, 1)
                    
                    elif rnn_output == "all_outputs_hidden":
                        self.concatenated_outputs = tf.concat(2, [self.bidirectional_rnn_output[0], self.bidirectional_rnn_output[1]])      
                        self.flattened_concatenated_rnn_outputs = tf.reshape(self.concatenated_outputs, [-1, rnn_cell_dim * 2])         #bidirectional -> * 2
                        self.rnn_output_hidden_layer = tf.contrib.layers.fully_connected(self.flattened_concatenated_rnn_outputs, num_outputs = rnn_cell_dim, activation_fn = tf.nn.relu)
                        self.reshaped_rnn_output_hidden_layer = tf.reshape(self.rnn_output_hidden_layer, [-1, max_command_len, rnn_cell_dim])
                        self.rnn_output = tf.reduce_sum(self.reshaped_rnn_output_hidden_layer, 1)

                    elif rnn_output == "direct_last_state":
                        _, self.bidirectional_rnn_state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, self.rnn_input, se

                        self.reference = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = 20, activation_fn = None)
                        self.reference_stacked = tf.stack([self.reference] * world_dimension, axis=2)   #[batch, 20, world_dimension]
                        self.world_reshaped = tf.reshape(self.world, [-1, 20, world_dimension])
                        self.multiple = tf.multiply(self.reference_stacked, self.world_reshaped)
                        self.location_by_reference = tf.reduce_sum(self.multiple, axis = 1)

                        self.location_by_direction = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = world_dimension, activation_fn = None)
                        self.predicted_location = tf.add(self.location_by_reference, self.location_by_direction)


                else:
                    if rnn_cell_type == "LSTM":
                        _, self.rnn_output_lstm_tuple = tf.nn.dynamic_rnn(rnn_cell, self.one_hot_words, sequence_length = self.command_lens)
                        self.rnn_output = self.rnn_output_lstm_tuple.c
                    else:
                        _, self.rnn_output = tf.nn.dynamic_rnn(rnn_cell, self.one_hot_words, sequence_length = self.command_lens)
                
                if use_world:
                    self.world_and_word = tf.concat(1, [self.world, self.rnn_output])
                    self.hidden_layer = tf.contrib.layers.fully_connected(self.world_and_word, num_outputs = hidden_dimension, activation_fn = tf.nn.relu)
                else:
                    self.hidden_layer = self.rnn_output
            
            elif network_type == "ffn":
                self.flattened_one_hot_words = tf.cast(tf.contrib.layers.flatten(self.one_hot_words), tf.float32)
                if use_world:
                    self.all_inputs = tf.concat(concat_dim = 1, values = [self.world, self.flattened_one_hot_words])
                    self.hidden_layer = tf.contrib.layers.fully_connected(self.all_inputs, num_outputs = hidden_dimension, activation_fn = tf.nn.relu)
                else:
                    self.hidden_layer = tf.contrib.layers.fully_connected(self.flattened_one_hot_words, num_outputs = hidden_dimension, activation_fn = tf.nn.relu)

            else:
                print(network_type)
                assert False
            
            ################################### DROPOUT OUTPUT ########################

            #self.hidden_layer = tf.scalar_mul(self.dropout_output_multiplier, tf.nn.dropout(self.hidden_layer, 1.0 - self.dropout_output_tensor))
            self.hidden_layer = tf.nn.dropout(self.hidden_layer, 1.0 - self.dropout_output_tensor)
            
            ################################### OUTPUT LAYER #########################

            if target == "source":
                self.logits = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = 20, activation_fn = None)
                self.predicted_source = tf.argmax(self.logits, 1)
                self.accuracy = tf.contrib.metrics.accuracy(tf.to_int32(self.predicted_source), self.source)
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.source)

            elif target == "location":
                if use_world:
                    self.predicted_location = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = world_dimension, activation_fn = None)
                else:
                    self.reference = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = 20, activation_fn = None)
                    self.reference_stacked = tf.stack([self.reference] * world_dimension, axis=2)   #[batch, 20, world_dimension]
                    self.world_reshaped = tf.reshape(self.world, [-1, 20, world_dimension])
                    self.multiple = tf.multiply(self.reference_stacked, self.world_reshaped)
                    self.location_by_reference = tf.reduce_sum(self.multiple, axis = 1)

                    self.location_by_direction = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = world_dimension, activation_fn = None)
                    self.predicted_location = tf.add(self.location_by_reference, self.location_by_direction)

                self.average_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.location - self.predicted_location), axis = 1)))
                self.loss = self.average_distance
            
            else:
                print(target)
                assert False

            self.training = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            if target == "source":
                self.train_summary = tf.summary.scalar("train/accuracy", self.accuracy)
                self.test_summary = tf.summary.scalar("test/accuracy", self.accuracy)
                self.dev_summary = tf.summary.scalar("dev/accuracy", self.accuracy)

            elif target == "location":
                self.train_summary = tf.summary.scalar("train/distance", self.average_distance)
                self.test_summary = tf.summary.scalar("test/distance", self.average_distance)
                self.dev_summary = tf.summary.scalar("dev/distance", self.average_distance)

            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.session.run(init)
    
#    
#    def rnn_layers(self, rnn_input, sequence_length, rnn_cell_type, rnn_cell_dim, layers, output):
#        if rnn_cell_type == "LSTM":
#            rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_cell_dim)
#
#        elif rnn_cell_type == "GRU":
#            rnn_cell = tf.nn.rnn_cell.GRUCell(rnn_cell_dim)
#
#        else:
#            raise ValueError("RNN cell type must be 'LSTM' or 'GRU'")
#        
#        for i in range(0, layers):
#            scope = "rnn_layer_" + str(self.rnn_layers)
#            rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, rnn_input, sequence_length = sequence_length, dtype = tf.float32, scope = scope)
#            rnn_input = tf.concat(2, [rnn_output[0], rnn_output[1]])
#
#        if output == "last_state":
#            if rnn_cell_type == "LSTM":
#                rnn_output = tf.concat(1, [rnn_state[0].c, rnn_state[1].c])
#            else:
#                rnn_output = tf.concat(1, self.bidirectional_rnn_state)
#
#        elif output == "outputs_sum":
#            concatenated_outputs = tf.concat(2, [bidirectional_rnn_output[0], bidirectional_rnn_output[1]])
#            rnn_output = tf.reduce_sum(concatenated_outputs, 1)
#        
#        elif output == "all_outputs:":
#            concatenated_outputs = tf.concat(
#            

        


    def get_command_lens(self, commands):
        command_lens = []
        for command in commands:
            command_lens.append(np.where(command == 1)[0][0] + 1)

        return command_lens
     

    def select_summary(self, dataset):
        if dataset == "dev":
            return self.dev_summary
        elif dataset == "test":
            return self.test_summary
        elif dataset == "train":
            return self.train_summary
        
        assert False


    def predict(self, commands, world, source_id, location, tags, dataset):
        predicted_location = None
        predicted_source = None
        feed_dict = {self.command : commands, self.command_lens : self.get_command_lens(commands), self.world : world, self.source : source_id, self.location : location,
                            self.dropout_input_tensor : 0.0, self.dropout_output_tensor : 0.0, self.dropout_input_multiplier : 1.0 - self.dropout_input, 
                            self.dropout_output_multiplier : 1.0 - self.dropout_output, self.tags : tags}

        if self.target == "location":
            predicted_location, summary = self.session.run([self.predicted_location, self.select_summary(dataset)], feed_dict)

        elif self.target == "source":
            predicted_source, summary = self.session.run([self.predicted_source, self.select_summary(dataset)], feed_dict)

        else:
            assert False

        self.summary_writer.add_summary(summary)

        return predicted_source, predicted_location

    
    def get_reference(self, commands, world, source_id, location, tags, dataset):
        feed_dict = {self.command : commands, self.command_lens : self.get_command_lens(commands), self.world : world, self.source : source_id, self.location : location,
                            self.dropout_input_tensor : 0.0, self.dropout_output_tensor : 0.0, self.dropout_input_multiplier : 1.0 - self.dropout_input, 
                            self.dropout_output_multiplier : 1.0 - self.dropout_output, self.tags : tags}

        if self.target == "location": 
            reference, location_reference, location_direction = self.session.run([self.reference, self.location_by_reference, self.location_by_direction], feed_dict)

        else:
            assert False

        return reference, location_reference, location_direction

    
    def train(self, commands, world, source_id, location, tags):
        feed_dict = {self.command : commands, self.command_lens : self.get_command_lens(commands), self.world : world, self.source : source_id, self.location : location,
                            self.dropout_input_tensor : self.dropout_input, self.dropout_output_tensor : self.dropout_output, self.dropout_input_multiplier : 1.0, self.dropout_output_multiplier : 1.0,
                            self.tags : tags}

        debug = False
        if debug == False:
            _, summary, loss  = self.session.run([self.training, self.select_summary("train"), self.loss], feed_dict)

        else:
            _, summary, loss, hidden, reference, location_reference, location_direction = self.session.run([self.training, self.select_summary("train"), self.loss, self.hidden_layer, self.reference, self.location_by_reference, self.location_by_direction], feed_dict)
                        
                       
        if debug:
            print("Reference: " + str(reference[0]))
            print("Location reference: " + str(location_reference[0]))
            print("Location direction: " + str(location_direction[0]))

        self.summary_writer.add_summary(summary)



