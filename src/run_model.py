
from dataset import Dataset
from scipy.spatial import distance
import copy
import random
import numpy as np
from network import Network
from benchmark_model import BenchmarkModel
import argparse
from database import Database
import pdb
import tensorflow as tf
import time
import os.path
import ast


def evaluate(model, dataset, epoch, dimension = 2):
    commands, worlds, sources, locations = dataset.get_all_data()
    predicted_sources, predicted_locations = model.predict(commands, worlds, sources, locations, dataset.dataset_name)
 
    source_accuracy = None
    location_distance = None

    if predicted_sources is not None:
        source_correct = 0
        for i in range(0, len(commands)):
            if sources[i] == predicted_sources[i]:
                source_correct += 1
            #else:
                #pdb.set_trace()
        
        source_accuracy = source_correct / float(len(commands))

    if predicted_locations is not None:    
        location_difference = 0
        for i in range(0, len(commands)):
            location_difference += distance.euclidean(predicted_locations[i], locations[i])
    
        location_distance = location_difference / float(len(commands))

    return (source_accuracy, location_distance, epoch)


def improvement(best_results, current_results):
    if current_results[0] is not None:
        return best_results[0] < current_results[0]

    if current_results[1] is not None:
        return best_results[1] > current_results[1]

    assert False


def print_results(results):
    print("Epoch: " + str(results[2]))
    if results[0] is not None:
        print("Source accuracy: " + str(results[0]))
    
    if results[1] is not None:
        print("Location distance: " + str(results[1]))
    
    print("")


def save(run_id, args, dev_result, test_result, start_time):
    db = Database()
    epoch = dev_result[2]
    if args.target == "source":
        dev_result = dev_result[0]
        test_result = test_result[0]
    else:
        dev_result = dev_result[1]
        test_result = test_result[1]

    seconds = (time.time() - start_time) * args.threads
    computation_time = str(int(seconds / 3600)) + ":" + str(int(seconds / 60) % 60)

    db_cols = ["run_id", "target", "model", "version", "dev_result", "test_result", "epoch", "hidden_dimension", "alpha", "rnn_cell_dim", "rnn_cell_type", "bidirectional", "dropout_input", "dropout_output", "batch_size", "use_world", "computation_time", "args"]
    
    args_to_delete = ["max_epochs", "test", "restore_and_test"]
    if model == "ffn":
        args_to_delete += ["rnn_cell_type", "rnn_cell_dim", "bidirectional"]
    
    args = vars(args)
    for to_delete in args_to_delete:
        if to_delete in args.keys():
            del args[to_delete]

    row = []
    
    for col in db_cols:
        if col in args.keys():
            row.append(args[col])
            del args[col]
        elif col in ["run_id", "dev_result", "test_result", "computation_time", "args"]:
            row.append(eval(col))
        else:
            row.append(None)

    db.insert("Results", row)
#
#    target = args.target
#    model = args.model
#    version = args.version
#    learning_rate = args.alpha
#    hidden_dimension = args.hidden_dimension
#    rnn_cell_dim = args.rnn_cell_dim
#    rnn_cell_type = args.rnn_cell_type
#    bidirectional = args.bidirectional
#    dropout_input = args.dropout_input
#    dropout_output = args.dropout_output
#    batch_size = args.batch_size
#
#    arguments = vars(args)
#    del arguments["target"]
#    del arguments["model"]
#    del arguments["alpha"]
#    del arguments["stop"]
#    del arguments["max_epochs"]
#    del arguments["test"]
#    del arguments["threads"]
#    del arguments["hidden_dimension"]
#    del arguments["restore_and_test"]
#    del arguments["version"]
#
#    if model == "ffn":
#        del arguments["rnn_cell_type"]
#        del arguments["rnn_cell_dim"]
#        del arguments["bidirectional"]
#
#    row = [run_id, target, model, version, dev_result, test_result, epoch, hidden_dimension, learning_rate, computation_time, str(arguments)]
#
#    db.insert("Results", row)
#

def get_run_id():
    if os.path.isfile(".id"):
        f = open(".id", "r+")
        run_id = int(f.read()) + 1
        f.seek(0)
        f.write(str(run_id))
        f.truncate()

    else:
        run_id = 0
        f = open(".id", "w")
        f.write(str(0))
    
    f.close()
    return run_id


def test_model(run_id):
    db = Database()
    db_output = db.get_all_rows("SELECT * FROM Results WHERE RunID = " + str(run_id))
    if len(db_output) != 1:
        raise ValueError("No data in database for run_id: " + str(run_id))

    row = list(db_output[0])
 
    version = row[3]
    
    dataset = Dataset("test", version)

    args = ast.literal_eval(row[-1])
    target = row[1]
    network_type = row[2]
    epoch = row[6]
    hidden_dimension = row[7]
    learning_rate = row[8]
    rnn_cell_dim = row[9]
    rnn_cell_type = row[10]
    bidirectional = bool(row[11])
    dropout_input = row[12]
    dropout_output = row[13]
    batch_size = row[14]
    use_world = row[15]

    if network_type == "ffn":
        rnn_cell_type = 'GRU'
        rnn_cell_dim = 0
        bidirectional = True

    model = Network(network_type, dataset.vocabulary_length(), hidden_dimension = hidden_dimension, run_id = run_id, learning_rate = learning_rate, target = target,
                        rnn_cell_dim = rnn_cell_dim, rnn_cell_type = rnn_cell_type, bidirectional = bidirectional, use_world = use_world, 
                        dropout_input = dropout_input, dropout_output = dropout_output, threads = 1)

    checkpoint = tf.train.get_checkpoint_state("checkpoints/" + str(run_id))
    model.saver.restore(model.session, checkpoint.model_checkpoint_path)

    test_results = evaluate(model, dataset, epoch)
    print_results(test_results)

    if test_results[0] != None:
        test_result = test_results[0]
    else:
        test_result = test_results[1]

    db.execute("UPDATE Results SET TestResult = " + str(test_result) + " WHERE RunID = " + str(run_id))
        


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="ffn", type=str, choices=["ffn", "rnn", "benchmark"], help="Model to use")
    parser.add_argument("--max_epochs", default=1000, type=int, help="Maximum number of epochs")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
    parser.add_argument("--stop", default=1, type=int, help="Number of epochs without improvement before stopping training")
    parser.add_argument("--hidden_dimension", default=128, type=int, help="Number of neurons in last hidden layer")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use")
    parser.add_argument("--alpha", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--rnn_cell_dim", default=512, type=int, help="Dimension of rnn cells.")
    parser.add_argument("--rnn_cell_type", default="GRU", type=str, choices=["LSTM", "GRU"], help="Type of rnn cell")
    parser.add_argument("--bidirectional", default=True, type=bool, help="Whether the RNN network is bidirectional")
    parser.add_argument("--target", default="source", type=str, choices=["source", "location"], help="Whether model should predict which block will be moved (source) or where it will be moved (location)")
    parser.add_argument("--test", default=False, type=bool, help="Test trained model on testing data")
    parser.add_argument("--restore_and_test", default=-1, type=int, help="Load model with given id and test it on test data")
    parser.add_argument("--use_world", default=False, type=bool, help="Whether model should use world state as input")
    parser.add_argument("--version", default=1, type=int, help="Which version of data to use")
    parser.add_argument("--dropout_input", default=0, type=float, help="Input dropout rate")
    parser.add_argument("--dropout_output", default=0, type=float, help="Output dropout rate")

    return parser.parse_args()


def main():
    random.seed(42)

    args = parse_arguments()
    
    if args.restore_and_test != -1:
        test_model(args.restore_and_test)
        return

    run_id = get_run_id()
    start_time = time.time()

    train_data = Dataset("train", args.version)
    dev_data = Dataset("dev", args.version)
    test_data = Dataset("test", args.version)

    if args.model in ["rnn", "ffn"]:
        model = Network(args.model, train_data.vocabulary_length(), hidden_dimension = args.hidden_dimension, run_id = run_id, learning_rate = args.alpha, target = args.target,
                        rnn_cell_dim = args.rnn_cell_dim, rnn_cell_type = args.rnn_cell_type, bidirectional = args.bidirectional, threads = args.threads, use_world = args.use_world,
                        dropout_input = args.dropout_input, dropout_output = args.dropout_output)

    elif args.model in ["benchmark"]:
        model = BenchmarkModel(args.version)
    
    epochs_without_improvement = 0
    best_results = (0.0, 1000000.0, -1)

    for epoch in range(args.max_epochs):
        train_data.next_epoch()
        
        while not train_data.epoch_end():
            commands, worlds, sources, locations = train_data.get_next_batch(args.batch_size)
            model.train(commands, worlds, sources, locations)

        current_results = evaluate(model, dev_data, epoch)
        print_results(current_results)
        
        if improvement(best_results, current_results):
            best_results = current_results
            epochs_without_improvement = 0
            if args.model != "benchmark":
                run_dir = "checkpoints/" + str(run_id)
                if not os.path.isdir(run_dir):
                    os.makedirs(run_dir)
                model.saver.save(model.session, run_dir + "/best.ckpt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == args.stop:
               break
    
    print("Best result:")
    print_results(best_results)

    if args.test:
        if args.model != "benchmark":
            best_net = tf.train.get_checkpoint_state("checkpoints/" + str(run_id))
            model.saver.restore(model.session, best_net.model_checkpoint_path)
        
        test_results = evaluate(model, test_data, best_results[2])

        print("Test results:")
        print_results(test_results)
    else:
        test_results = (None, None, None)


    save(run_id, args, best_results, test_results, start_time)


        


main()       
            
 
