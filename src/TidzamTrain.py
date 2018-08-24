from __future__ import print_function

import sys, optparse
import numpy as np
import shutil
import time
import math
import os

import vizualisation as vizu
import TidzamDatabase as database
from App import App
import json
from display_board import DisplayBoard

'''
def build_command_from_conf(conf_file):
    try:
        with open(conf_file) as json_file:
            conf_data = json.load(json_file)
    except:
        App.log(0 , "There isn't any valide json file")
        return []

    build_cmd_line = []
    for key , value in conf_data.items():
        if is_primitive(value) or isinstance(value , int):
            build_cmd_line.append("--{}={}".format(key , value))
    return build_cmd_line
'''
'''
primitive_list = [str ,int ,float, bool]
def is_primitive(var):
    for primitive_var_type in primitive_list:
        if isinstance(var , primitive_var_type):
            return True
    return False
'''

def overwrite_conf_with_opts(conf_data , opts , default_values_dic):
    for key , value in default_values_dic.items():
        if key not in conf_data:
            conf_data[key] = value
    for key , value in opts.items():
        if value is not None or key not in conf_data:
            conf_data[key] = value


def push_new_memory(outputs_memory , labels_memory , new_outputs , new_labels , memory_size = 10000):
    if outputs_memory is None:
        new_outputs_memory = new_outputs
        new_labels_memory = new_labels
    else:
        new_outputs_memory = np.concatenate( (new_outputs , outputs_memory)  , axis = 0 )[:memory_size , :]
        new_labels_memory = np.concatenate( (new_labels , labels_memory)  , axis = 0 )[:memory_size , :]

    return new_outputs_memory , new_labels_memory

if __name__ == "__main__":
    from tensorflow.contrib.tensorboard.plugins import projector
    import tensorflow as tf

    App.log(0, "TensorFlow "+ tf.__version__)

    ###################################
    ### Console Parameters
    ###################################
    usage="TidzamTrain.py --dataset-train=mydataset --dnn=models/model.py --out=save/ [OPTIONS]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-d", "--dataset-train",
        action="store", type="string", dest="dataset_train",
        help='Define the dataset to train.')

    parser.add_option("-t", "--dataset-test",
        action="store", type="string", dest="dataset_test",
        help='Define the dataset for evaluation.')

    parser.add_option("-o", "--out",
        action="store", type="string", dest="out",
        help='Define output folder to store the neural network and trains.')

    parser.add_option("--dnn",
        action="store", type="string", dest="dnn",
        help='DNN model to train (Default: ).')
    ###
    parser.add_option("--training-iterations",
        action="store", type="int", dest="training_iters",
        help='Number of training iterations (Default: 20000 iterations).')

    parser.add_option("--testing-step",
        action="store", type="int", dest="testing_iterations",
        help='Number of training iterations between each testing step (Default: 10).')

    parser.add_option("--batchsize",
        action="store", type="int", dest="batch_size",
        help='Size of the training batch (Default:64).')

    parser.add_option("--learning-rate",
        action="store", type="float", dest="learning_rate",
        help='Learning rate (default: 0.001).')
    ###
    parser.add_option("--stats-step",
        action="store", type="int", dest="STATS_STEP",
        help='Step period to compute statistics, embeddings and feature maps (Default: 10).')

    parser.add_option("--nb-embeddings",
        action="store", type="int", dest="nb_embeddings",
        help='Number of embeddings to compute (default: 50)..')
    ###
    parser.add_option("--job-type",
        action="store", type="string", dest="job_type",
        help='Selector the process job: ps or worker (default:worker).')

    parser.add_option("--task-index",
        action="store", type="int", dest="task_index",
        help='Provide the task index to execute (default:0).')

    parser.add_option("--workers",
        action="store", type="string", dest="workers",
        help='List of workers (worker1.mynet:2222,worker2.mynet:2222, etc).')

    parser.add_option("--ps",
        action="store", type="string", dest="ps",
        help='List of parameter servers (ps1.mynet:2222,ps2.mynet:2222, etc).')

    parser.add_option("--cutoff-up",
        action="store", type="string", dest="cutoff_up",
        help='Pixel cutoff on frequency axis high value.')

    parser.add_option("--cutoff-down",
        action="store", type="string", dest="cutoff_down",
        help='Pixel cutoff on frequency axis low value.')

    parser.add_option("--conf-file", action="store", type="string", dest="conf_file" ,
        default="" , help="json file holding the data necessary about class access path and type")
    ###

    default_values_dic = {"dataset_train" : "" ,"out" : "/tmp/tflearn_logs" , "dnn" : "default" , "training_iters" : 20000,"testing_iterations" : 10,
                            "batch_size" : 64, "learning_rate" : 0.001, "STATS_STEP" : 20, "nb_embeddings" : 50, "task_index" : 0, "workers" : "localhost:2222","ps": "",
                            "job_type" : "worker", "cutoff_down":20, "cutoff_up":170 }

    (opts, args) = parser.parse_args()
    opts = vars(opts)

    try:
        with open(opts["conf_file"]) as json_file:
            conf_data = json.load(json_file)
    except:
        App.log(0 , "There isn't any valide json file")
        exit()

    overwrite_conf_with_opts(conf_data , opts , default_values_dic)

    ###################################
    # Cluster configuration
    ###################################
    ps      = conf_data["ps"].split(",")
    workers = conf_data["workers"].split(",")
    cluster  = {"worker":workers}
    if ps[0] != "":
        cluster["ps"] = ps
    cluster = tf.train.ClusterSpec(cluster)

    # start a server for a specific task
    server = tf.train.Server(cluster,   job_name=conf_data["job_type"],
                                        task_index=conf_data["task_index"])

    if conf_data["job_type"] == "ps":
        App.log(0, "Parameter server " + ps[conf_data["task_index"]]+ " started.")
        server.join()
    elif conf_data["job_type"] != "worker":
        App.log(0, "Bad argument in job name [ps | worker]")
        sys.exit(0)

    App.log(0, "Worker " + workers[conf_data["task_index"]]+ " started")

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.25,
        allow_growth=True
        )

    config = tf.ConfigProto(
            intra_op_parallelism_threads=4,
            inter_op_parallelism_threads=4,
            gpu_options=gpu_options,
#            log_device_placement=True,
            allow_soft_placement=True
            )


    ###################################
    # Load the data
    ###################################    '''
    dataset      = database.Dataset(conf_data["dataset_train"] , conf_data=conf_data)
    if conf_data["dataset_test"]:
        dataset_test = database.Dataset(conf_data["dataset_test"], class_file=conf_data)
    App.log(0, "Sample size: " + str(dataset.size[0]) + 'x' + str(dataset.size[1]))
    conf_data["size"]   = dataset.size

    ###################################
    # Between-graph replication
    ###################################
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % conf_data["task_index"],
        cluster=cluster)):

        global_step  = tf.train.get_or_create_global_step()
        writer_train = tf.summary.FileWriter(conf_data["out"]+"/model/train/")
        writer_test  = tf.summary.FileWriter(conf_data["out"]+"/model/test/")

        ###################################
        # Build graphs
        ###################################
        App.log(0, "Loading DNN model from:  " + conf_data["dnn"])
        sys.path.append('./')
        exec("import "+os.path.dirname(conf_data["dnn"])+"."+os.path.basename(conf_data["dnn"]).replace(".py","")+" as model")

        net = eval("model.DNN(conf_data)")

        ## Generate summaries
        with tf.name_scope('Summaries'):
            summaries = vizu.Summaries(net, conf_data)
            outputs_test_memory = None
            outputs_train_memory = None
            labels_test_memory = None
            labels_train_memory = None

            ## Construct filter images
            with tf.name_scope('Visualize_filters'):
                summaries.build_kernel_filters_summaries(net.show_kernel_map)

        with tf.variable_scope("embeddings"):
            conf_data["nb_embeddings"] = min(conf_data["nb_embeddings"], dataset.data.shape[0])
            proj      = projector.ProjectorConfig()
            embed_train     = vizu.Embedding("OUT_train", net.input, net.out, net.keep_prob, proj, conf_data["nb_embeddings"], conf_data["out"])
            embed_test      = vizu.Embedding("OUT_test",  net.input, net.out, net.keep_prob, proj, conf_data["nb_embeddings"], conf_data["out"])
            projector.visualize_embeddings(writer_test, proj)
            projector.visualize_embeddings(writer_train, proj)

        App.log(0, "Generate summaries graph.")
        merged = tf.summary.merge_all()

        with tf.name_scope('Trainer'):
            train_op = tf.train.AdagradOptimizer(conf_data["learning_rate"]).minimize(net.cost, global_step=global_step)
            hooks=[tf.train.StopAtStepHook(last_step=conf_data["training_iters"])]

        ###################################
        # Initialize Printing_board
        ###################################
        display_board = DisplayBoard(conf_data["classes_list"] , 20, 60)

        display_board.add_new_row("Precision_Train")
        display_board.add_new_row("Recall_Train")
        display_board.add_new_row("F1_score_Train")

        display_board.add_new_row("Precision_Test")
        display_board.add_new_row("Recall_Test")
        display_board.add_new_row("F1_score_Test")

        ###################################
        # Start the session
        ###################################
        if conf_data["task_index"] != 0:
            App.log(0, "Waiting for the master worker.")


        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(conf_data["task_index"] == 0),
                                               checkpoint_dir=conf_data["out"]+"/model/",
                                               hooks=hooks,
                                               config=config) as sess:
                App.ok(0, "Training is starting.")
                writer_train.add_graph(sess.graph)
                writer_test.add_graph(sess.graph)

                # Save the model and conf file
                shutil.copyfile(conf_data["dnn"], conf_data["out"] + "/model.py")
                with open(conf_data["out"] + "/conf.json", 'w') as outfile:
                    json.dump(conf_data, outfile)

                while not sess.should_stop():
                    #App.log(0, "---")
                    start_time = time.time()

                    ################### TRAINING
                    batch_x, batch_y    = dataset.next_batch(batch_size=conf_data["batch_size"])

                    _, step = sess.run( [train_op, global_step],
                        feed_dict={ net.input: batch_x, net.labels: batch_y, net.keep_prob: 0.5 })

                    if step % conf_data["STATS_STEP"] == 0:
                        run_options         = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata        = tf.RunMetadata()

                        _  , net_output_train , cost_train = sess.run(
                            [train_op, net.output , net.cost],
                            feed_dict={ net.input: batch_x, net.labels: batch_y, net.keep_prob: 0.25 },
                            options=run_options, run_metadata=run_metadata)

                        outputs_train_memory , labels_train_memory = push_new_memory(outputs_train_memory , labels_train_memory , net_output_train , batch_y)

                        precision_train , recall_train , f1_score_train , summary_train = sess.run(
                            [summaries.precision , summaries.recall , summaries.f1_score, merged ],
                            feed_dict={
                                    net.input: batch_x,
                                    net.labels: batch_y,
                                    net.keep_prob: 0.25,
                                    summaries.outputs: outputs_train_memory,
                                    summaries.labels: labels_train_memory,
                                    })

                        embed_train.evaluate(
                                    batch_x[:conf_data["nb_embeddings"],:],
                                    batch_y[:conf_data["nb_embeddings"],:],
                                    session=sess,
                                    dic=dataset.conf_data["classes_list"])

                        #summaries.evaluate(batch_x, batch_y, sess)
                        writer_train.add_run_metadata(run_metadata, 'step%d' % step)
                        writer_train.add_summary(summary_train, step)
                    else:
                        cost_train = 0
                        precision_train = None

                    ################### TESTING
                    if step % conf_data["testing_iterations"] == 0:
                        if conf_data["dataset_test"]:
                            batch_test_x, batch_test_y  = dataset_test.next_batch(batch_size=conf_data["batch_size"])
                        else:
                            batch_test_x, batch_test_y  = dataset.next_batch(batch_size=conf_data["batch_size"], testing=True)

                        net_output_test , cost_test = sess.run(
                            [net.output, net.cost],
                            feed_dict={net.input: batch_test_x,net.labels: batch_test_y, net.keep_prob: 1.0 })

                        outputs_test_memory , labels_test_memory = push_new_memory(outputs_test_memory , labels_test_memory , net_output_test , batch_test_y)

                        precision_test , recall_test , f1_score_test, summary_test = sess.run(
                            [summaries.precision , summaries.recall , summaries.f1_score , merged],
                            feed_dict={
                                    net.input: batch_test_x,
                                    net.labels: batch_test_y,
                                    net.keep_prob: 1.0,
                                    summaries.outputs: outputs_test_memory,
                                    summaries.labels: labels_test_memory,
                                    })

                        if step % conf_data["STATS_STEP"] == 0:
                            embed_test.evaluate(
                                        batch_test_x[:conf_data["nb_embeddings"],:],
                                        batch_test_y[:conf_data["nb_embeddings"],:],
                                        session=sess,
                                        dic=dataset.conf_data["classes_list"])
                        summaries.evaluate(batch_test_x, batch_test_y, sess)
                        writer_test.add_summary(summary_test, step)
                    else:
                        cost_test = 0
                        precision_test = None

                    if precision_test is not None and precision_train is not None:
                        display_board.update_row_values("Precision_Train" , precision_train)
                        display_board.update_row_values("Precision_Test" , precision_test)
                        display_board.update_row_values("Recall_Train" , recall_train)
                        display_board.update_row_values("Recall_Test" , recall_test)
                        display_board.update_row_values("F1_score_Train" , f1_score_train)
                        display_board.update_row_values("F1_score_Test" , f1_score_test)
                        display_board.display()

                        App.log(0,  "\033[1;37m Step {0} - \033[0m {1:.2f} sec  | train - cost \033[32m{2:.3f}\033[0m | test - cost \033[32m{3:.3f}\033[0m |".format(
                                        step,
                                        time.time() - start_time,
                                        cost_train,
                                        cost_test,
                                         ))
                        '''
                        App.log(0 , "\x1B[31mprecision\x1B[0m : \ntrain - acc \033[32m{0}\033[0m \ntest - acc \033[32m{1}\033[0m".format(np.array_str(precision_train, max_line_width=1000000) ,
                                                                                                                                                     np.array_str(precision_test, max_line_width=1000000)) )
                        App.log(0 , "\x1B[31mrecall\x1B[0m : \ntrain - acc \033[32m{0}\033[0m \ntest - acc \033[32m{1}\033[0m".format(np.array_str(recall_train, max_line_width=1000000) ,
                                                                                                                                                     np.array_str(recall_test, max_line_width=1000000)) )
                        App.log(0 , "\x1B[31mf1_score\x1B[0m : \ntrain - acc \033[32m{0}\033[0m \ntest - acc \033[32m{1}\033[0m".format(np.array_str(f1_score_train, max_line_width=1000000) ,
                                                                                                                                                     np.array_str(f1_score_test, max_line_width=1000000)) )

                        App.log(0 , f1_score_train.shape)
                        exit()
                        '''
                        #App.log(0 , "Ground truth accuracy by class train : \ntrain - acc \033[32m{0}\033[0m \ntest - acc \033[32m{1}\033[0m".format(np.array_str(class_accuracy_train, max_line_width=1000000) ,
                        #                                                                                                                             np.array_str(class_accuracy_test, max_line_width=1000000)) )
                        #App.log(0 , "Some outputs examples : \ntrain - outputs examples\n \033[32m{0}\033[0m \ntest - outputs examples\n \033[32m{1}\033[0m".format(np.array_str(outputs_examples_train, max_line_width=1000000) ,
                        #
                        #                                                                                                                                  np.array_str(outputs_examples_test, max_line_width=1000000)))
