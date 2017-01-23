
import tensorflow as tf
from setting import max_command_len
import numpy as np
import pdb

class Network:
    def __init__(self, network_type, vocabulary_length, run_id, learning_rate, target, rnn_cell_type, rnn_cell_dim, bidirectional, hidden_dimension, use_world threads, world_dimension = 2):
        self.target = target

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
            self.world = tf.placeholder(tf.float32, [None, 20 * world_dimension])     #[batch, block_coordinates]
            self.source = tf.placeholder(tf.int32, [None])
            self.location = tf.placeholder(tf.float32, [None, world_dimension])

            self.one_hot_words = tf.one_hot(self.command, vocabulary_length)
            
            if network_type == "rnn":
               
                if bidirectional:
                    _, self.bidirectional_rnn_state = tf.nn.bidirectional_dynamic_rnn(rnn_cell, rnn_cell, self.one_hot_words, sequence_length = self.command_lens, dtype = tf.float32)
                    self.rnn_output = tf.concat(1, self.bidirectional_rnn_state)

                else:
                    _, self.rnn_output = tf.nn.dynamic_rnn(rnn_cell, self.one_hot_words, sequence_length = self.command_lens)

                #self.world_and_word = tf.concat(1, [self.world, self.rnn_output])
                #self.hidden_layer = tf.contrib.layers.fully_connected(self.world_and_word, num_outputs = hidden_dimension, activation_fn = tf.nn.relu)
                self.hidden_layer = self.rnn_output
            
            elif network_type == "ffn":
                self.flattened_one_hot_words = tf.cast(tf.contrib.layers.flatten(self.one_hot_words), tf.float32)
                #self.all_inputs = tf.concat(concat_dim = 1, values = [self.world, self.flattened_one_hot_words])
                #self.hidden_layer = tf.contrib.layers.fully_connected(self.all_inputs, num_outputs = hidden_dimension, activation_fn = tf.nn.relu)
                self.hidden_layer = tf.contrib.layers.fully_connected(self.flattened_one_hot_words, num_outputs = hidden_dimension, activation_fn = tf.nn.relu)

            else:
                print(network_type)
                assert False
 
            
            if target == "source":
                self.logits = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = 20, activation_fn = None)
                self.predicted_source = tf.argmax(self.logits, 1)
                self.accuracy = tf.contrib.metrics.accuracy(tf.to_int32(self.predicted_source), self.source)
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.source)

            elif target == "location":
                self.predicted_location = tf.contrib.layers.fully_connected(self.hidden_layer, num_outputs = world_dimension, activation_fn = None)
                #self.loss = tf.reduce_mean(tf.square(self.location - self.predicted_location))
                self.average_distance = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.location - self.predicted_location), axis = 1)))
                self.loss = self.average_distance
            
            else:
                print(target)
                assert False

            self.training = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.dataset_name = tf.placeholder(tf.string, [])
            if target == "source":
                self.summary = tf.scalar_summary(self.dataset_name + "/accuracy", self.accuracy)

            elif target == "location":
                self.summary = tf.scalar_summary(self.dataset_name + "/distance", self.average_distance)

            self.saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            self.session.run(init)
    

    def get_command_lens(self, commands):
        command_lens = []
        for command in commands:
            command_lens.append(np.where(command == 1)[0][0] + 1)

        return command_lens
     

    def predict(self, commands, world, source_id, location, dataset):
        predicted_location = None
        predicted_source = None
        feed_dict = {self.command : commands, self.command_lens : self.get_command_lens(commands), self.world : world, self.source : source_id, self.location : location, self.dataset_name : dataset}
        if self.target == "location": 
            predicted_location, summary = self.session.run([self.predicted_location, self.summary], feed_dict)

        elif self.target == "source":
            predicted_source, summary = self.session.run([self.predicted_source, self.summary], feed_dict)

        else:
            assert False

        self.summary_writer.add_summary(summary)

        return predicted_source, predicted_location

    
    def train(self, commands, world, source_id, location):
        feed_dict = {self.command : commands, self.command_lens : self.get_command_lens(commands), self.world : world, self.source : source_id, self.location : location, self.dataset_name : "train"}
        _, summary, loss = self.session.run([self.training, self.summary, self.loss], feed_dict)

        self.summary_writer.add_summary(summary)



