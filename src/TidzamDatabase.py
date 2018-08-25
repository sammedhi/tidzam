from __future__ import print_function
from __future__ import division

import sys, os
import numpy as np
import optparse
import random
import math
import copy
from matplotlib import pyplot as plt
import glob
import time
import re

import multiprocessing as mp
import atexit

from scipy import signal
import soundfile as sf
import sounddevice as sd
import traceback

from App import App
import json

###################################SOUND PROCESS##########################################


#This function blend a sound inside a given background and repeat the process for a given number of time
#to produce multiple samples with the same sound at different position
def blend_sound_to_background_severals_positions(sound_data , ambiant_sound_data , number_of_instance):
    new_outputs = []
    for i in range(number_of_instance):
        new_outputs.append(blend_sound_to_background(sound_data , ambiant_sound_data))
    return new_outputs

#This function blend a sound inside a given background at a random position
def blend_sound_to_background(sound_data , ambiant_sound_data):
    volume_factor = np.max(sound_data)

    sound_data = sound_data / np.max(sound_data) * volume_factor
    ambiant_sound_data = ambiant_sound_data / np.max(ambiant_sound_data) * volume_factor

    sound_data_position = int(random.randint(0 , len(ambiant_sound_data) - len(sound_data)) )

    if sound_data_position < 0:
        raise Exception("Corrupted Data")

    signal_sum = ambiant_sound_data[:]
    for i in range(len(sound_data)):
        try:
            signal_sum[sound_data_position + i] += sound_data[i]
        except:
            signal_sum[sound_data_position + i] += sound_data[i,0]

    mixed_signal = signal_sum / np.max(signal_sum) * volume_factor
    return mixed_signal

def blend_multiple_sound_to_background(sounds_data , ambiant_sound_data):
    for sound_data in sounds_data:
        ambiant_sound_data = blend_sound_to_background(sound_data , ambiant_sound_data)
    return ambiant_sound_data

def convert_to_monochannel(input):
    return input if len(input.shape) <= 1 else input[: , 0]

###################################SOUND PROCESS##########################################

def get_spectrogram(data, samplerate, channel=0,  show=False, cutoff=[20,170]):
    plt.ion()
    fs, t, Sxx = signal.spectrogram(data, samplerate, nfft=1024, noverlap=128)
    # editor between 1-8 Khz
    if cutoff is not []:
        Sxx = Sxx[[x for x in range( max(0,cutoff[0]) , min(cutoff[1], len(Sxx)) )], :]*1000
        fs = fs[[x for x in range( max(0,cutoff[0]) , min(cutoff[1], len(Sxx)) )]]
    # Normalize and cutoff
    Sxx = np.maximum(Sxx/np.max(Sxx), np.ones((Sxx.shape[0], Sxx.shape[1]))*0.01)

    if show is True:
        plt.figure(channel, figsize=(7, 7))
        plt.pcolormesh(t, fs, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        plt.pause(0.1)
        sd.play(data, samplerate)
        time.sleep(0.5)

    size = [Sxx.shape[0], Sxx.shape[1]]
    Sxx = np.reshape(Sxx, [1, Sxx.shape[0]*Sxx.shape[1]] )
    return fs, t, Sxx, size

def play_spectrogram_from_stream(file, show=False, callable_objects = [], overlap = 0, cutoff=[20,170]):

    with sf.SoundFile(file, 'r') as f:
        while f.tell() < len(f):
            data = f.read(int(f.samplerate/2))

            for i in range(0,f.channels):
                if f.channels > 1:
                    fs, t, Sxx, size = get_spectrogram(data[:,i], f.samplerate, i,  show=show, cutoff=cutoff)
                else:
                    fs, t, Sxx, size  = get_spectrogram(data, f.samplerate, i, show=show, cutoff=cutoff)

                if i == 0:
                    Sxxs = Sxx
                    fss = fs
                    ts = t
                else:
                    Sxxs = np.concatenate((Sxxs, Sxx), axis=0)
                    fss = np.concatenate((fss, fs), axis=0)
                    ts = np.concatenate((ts, t), axis=0)

            for obj in callable_objects:
                obj.run(Sxxs, fss, ts, [data, f.samplerate], overlap=overlap)

            f.seek(int(-int(f.samplerate/2)*overlap), whence=sf.SEEK_CUR)

        return Sxx, t, fs, size

def play_spectrogram_from_stream_data(data , samplerate , channels , show=False, callable_objects = [], overlap = 0):
    idx = 0
    while idx < len(data):
        trunc_data = data[ idx : idx + samplerate // 2]

        for i in range(0,channels):
            if channels > 1:
                fs, t, Sxx, size = get_spectrogram(trunc_data[:,i], samplerate, i,  show=show)
            else:
                fs, t, Sxx, size  = get_spectrogram(trunc_data, samplerate, i, show=show)

            if i == 0:
                Sxxs = Sxx
                fss = fs
                ts = t
            else:
                Sxxs = np.concatenate((Sxxs, Sxx), axis=0)
                fss = np.concatenate((fss, fs), axis=0)
                ts = np.concatenate((ts, t), axis=0)

            for obj in callable_objects:
                obj.run(Sxxs, fss, ts, [trunc_data, samplerate], overlap=overlap)

            idx += samplerate // 2 + int(-int(samplerate/2)*overlap)

    return Sxx, t, fs, size

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

class Dataset:
    def __init__(self, name="/tmp/dataset",conf_data = None, p=0.9, max_file_size=1000, split=0.9, cutoff=[20,170]):
        self.cur_batch = 0
        self.size           = None
        self.max_file_size  = max_file_size
        self.name           = name

        self.fileID         = 0
        self.files_count    = 0
        self.batchIDfile    = 0
        self.batchfile      = []
        self.cur_batch      = 0

        self.data           = []
        self.labels         = []
        self.batch_size     = conf_data["batch_size"]

        self.mode                  = None
        self.thread_count_training = 3
        self.thread_count_testing  = 1
        self.threads               = []
        self.queue_training        = None
        self.queue_maxsize         = 20
        self.split                 = split

        self.conf_data = conf_data
        self.class_tree = None

        self.cutoff = cutoff
        if(self.conf_data["cutoff_up"] is not None and self.conf_data["cutoff_down"] is not None ):
            self.cutoff = [ int(conf_data["cutoff_down"]), int(conf_data["cutoff_up"]) ]

        self.cutoff = cutoff
        if(self.conf_data["cutoff_up"] is not None and self.conf_data["cutoff_down"] is not None ):
            self.cutoff = [ int(conf_data["cutoff_down"]), int(conf_data["cutoff_up"]) ]

        atexit.register(self.exit)

        self.load(self.name)

    def exit(self):
        App.log(0, " Exit.")
        for t in self.threads:
            t. terminate()

    def build_labels_dic(self):
        self.conf_data["classes"] = self.sort_classes(self.conf_data["classes"])
        self.conf_data["classes_list"] = self.class_flattening(self.conf_data["classes"])

    ##########################################
    # Flatten the "classes" inside the conf_data json file to get a list of class instead
    # of a comlex object .
    # Here a simple example :
    # [ {"name" : "class1" , "classes" : ["subc1" , "subc2"]} , "class2"]
    #                   =========>
    # [ "class1" , "subc1" , "subc2" , "class2" ]
    ###########################################
    def class_flattening(self , cl_list):
        output = []
        for obj in cl_list:
            if isinstance(obj , str):
                output.append(obj)
            else:
                output.append(obj["name"])
                sub_output = self.class_flattening(obj["classes"])
                for cl in sub_output:
                    output.append(cl)
        return output

    ##########################################
    # Sort the "classes" inside the conf_data json file .
    # Here a simple example :
    # [ {"name" : "b" , "classes" : ["s" , "d" , "l"]} , "a"]
    #                   =========>
    # [ "a" , {"name" : "b" , "classes" : ["d" , "l" , "s"]}]
    ###########################################
    def sort_classes(self , list_cl):
        sorted_list = []
        for cl in list_cl:
            if not isinstance(cl , str):
                cl["classes"] = self.sort_classes(cl["classes"])
            inserted = False
            for index , sorted_cl in enumerate(sorted_list):
                name_cl = cl if isinstance(cl , str) else cl["name"]
                name_sorted_cl = sorted_cl if isinstance(sorted_cl , str) else sorted_cl["name"]
                if name_cl < name_sorted_cl:
                    sorted_list.insert(index , cl)
                    inserted = True
                    break
            if not inserted:
                sorted_list.append(cl)
        return sorted_list


    def build_output_vector(self , class_name , inheritence_dic):
        label = np.zeros((1,len(self.out_labels)))
        for cl_index in inheritence_dic[class_name]:
            label[0 , cl_index] = 1
        return label


    def load(self, folder ):
        self.name   = folder
        ctx = mp.get_context('spawn')
        self.queue_training  = ctx.Queue(self.queue_maxsize)
        self.queue_testing   = ctx.Queue(self.queue_maxsize)

        # Build class dictionnary
        self.build_labels_dic()
        self.out_labels = self.conf_data["classes_list"]

        App.log(0 ,"trained classes are : " + str(self.conf_data["classes"]))

        # Extract file for training and testing
        if self.split == None:
            App.log(0, "You must specify the attribute --split for the proportion of testing sample")
            return

        self.files_training = {}
        self.files_testing  = {}

        if "object" in self.conf_data:
            cl_paths_list = [np.array(glob.glob(object["path"] + "*/**/*.wav", recursive=True)) for object in self.conf_data["object"]]
            cl_names = [object["name"] for object in self.conf_data["object"]]
            cl_type = [object["type"] for object in self.conf_data["object"]]
        else:
            cl_paths_list = [np.array(glob.glob(self.name + "/" + cl + "*/**/*.wav", recursive=True)) for cl in self.conf_data["classes_list"]]
            cl_names = self.conf_data["classes_list"]

        raw, time, freq, self.size   = play_spectrogram_from_stream(cl_paths_list[0][0], cutoff=self.cutoff)

        #if self.conf_data is None:
        for cl , paths in zip(cl_names , cl_paths_list):
            files_cl = paths
            idx = np.arange(len(files_cl))
            np.random.shuffle(idx)
            self.files_training[cl] = files_cl[ idx[:int(len(idx)*self.split)] ]
            self.files_testing[cl]  = files_cl[ idx[int(len(idx)*self.split):] ]
            App.log(0, "training / testing datasets for " + cl + ": " + str(len(self.files_training[cl])) + " / " +str(len(self.files_testing[cl]))+" samples" )

        # Start the workers
        for i in range(self.thread_count_training):
            t = ctx.Process(target=self.build_batch_onfly,
                    args=(self.queue_training, self.files_training, self.batch_size , True))
            t.start()
            self.threads.append(t)

        for i in range(self.thread_count_testing):
            t = ctx.Process(target=self.build_batch_onfly,
                    args=(self.queue_testing, self.files_testing, self.batch_size))
            t.start()
            self.threads.append(t)

        while self.queue_training.empty():
            pass
        self.data, self.labels = self.queue_training.get()

    def next_batch(self, batch_size=128, testing=False):
        if testing is False:
            if self.queue_training.qsize() == 0:
                App.log(0, "Next batch size on fly is waiting (queue empty) - train")
            while self.queue_training.empty():
                pass
            return self.queue_training.get()
        else:
            if self.queue_testing.qsize() == 0:
                App.log(0, "Next batch size on fly is waiting (queue empty). - test")
            while self.queue_testing.empty():
                pass
            return self.queue_testing.get()

    def build_inheritence_dic(self , classes , dic = dict() , parent_indexes = []):
        for cl in classes:
            if isinstance(cl , str):
                cl_index = [index for index , cl_ in enumerate(self.conf_data["classes_list"]) if cl_ == cl]
                dic[cl] = parent_indexes + cl_index
            else:
                cl_index = [index for index , cl_ in enumerate(self.conf_data["classes_list"]) if cl_ == cl["name"]]
                dic[cl["name"]] = parent_indexes + cl_index

                self.build_inheritence_dic(cl["classes"] ,
                                           dic ,
                                           parent_indexes + cl_index)
        return dic


    def build_batch_onfly(self, queue, files, batch_size=64 , is_training = False , file_chunk_size = 320):
        #List all the ambiant class
        ambiant_cl = [object["name"] for object in self.conf_data["object"] if object["type"] == "background"]
        type_dictionnary = dict()
        augmentation_dictionnary = dict()
        inheritence_dic = self.build_inheritence_dic(self.conf_data["classes"])
        file_chunk_size += batch_size - (file_chunk_size % batch_size)

        for cl in self.conf_data["object"]:
            type_dictionnary[cl["name"]] = cl["type"]
            if "is_augmented" in cl and cl["is_augmented"]:
                augmentation_dictionnary[cl["name"]] = cl["is_augmented"]

        while True:
            samples = self.pick_samples(files , self.conf_data["classes"] , file_chunk_size)
            number_batch = len(samples) // batch_size
            for batch_id in range(number_batch):
                data = []
                labels = []
                batch_samples = samples[batch_id * batch_size : (batch_id + 1) * batch_size]

                for sample in batch_samples:
                    cl = sample[0]
                    files_cl = files[cl]
                    id = sample[1]
                    #for each picked sample -> process
                    #for id in idx:
                    sound_data , samplerate = sf.read(files_cl[id])
                    sound_data = sound_data if len(sound_data.shape) <= 1 else convert_to_monochannel(sound_data)

                    if is_training and type_dictionnary[cl] == "content" and cl in augmentation_dictionnary:
                        ambiant_file = random.choice(files[random.choice(ambiant_cl)])
                        ambiant_sound , samplerate = sf.read(ambiant_file)
                        ambiant_sound = ambiant_sound if len(ambiant_sound.shape) <= 1 else convert_to_monochannel(ambiant_sound)
                        try:
                            sound_data = blend_sound_to_background(sound_data , ambiant_sound)
                        except:
                            App.Log(0 , "One of these 2 files are corrupted (or probably both) : " , files_cl[id] , " , " , ambiant_file)


                    try:
                        raw, time, freq, size   = play_spectrogram_from_stream(files_cl[id],cutoff=self.cutoff)
                        raw                     = np.nan_to_num(raw)
                        raw                     = np.reshape(raw, [1, raw.shape[0]*raw.shape[1]])
                        label                   = self.build_output_vector(cl ,inheritence_dic)

                        try:
                            data = np.concatenate((data, raw), axis=0)
                            labels = np.concatenate((labels, label), axis=0)
                        except:
                            data   = raw
                            labels = label

                    except Exception as e :
                        App.log(0, "Bad file" + str(e))
                        traceback.print_exc()

                #Shuffle the final batch
                idx = np.arange(data.shape[0])
                np.random.shuffle(idx)
                data   = data[idx,:]
                labels = labels[idx,:]

                data   = data[:batch_size,:]
                labels = labels[:batch_size,:]

                while self.queue_training.full():
                    pass

                queue.put([data, labels])


            '''
            count = math.ceil(batch_size / len(self.conf_data["classes_list"]))
            data = []
            labels = []

            for i , cl in enumerate(self.conf_data["classes_list"]):
                #pick random sample (only get the indexes)
                try:
                    files_cl = files[cl]
                except:
                    # TODO
                    # Comment
                    continue

                idx = np.arange(len(files_cl))
                np.random.shuffle(idx)
                idx = idx[:count]
            '''


    def pick_samples(self , files , cl_list , size , parent_cl=None):
        output_files = []
        cl_list = cl_list + ([] if parent_cl is None else [parent_cl])
        count = math.ceil(size / len(cl_list) )

        for i , cl in enumerate(cl_list):
            if isinstance(cl , str):
                try:
                    files_cl = files[cl]
                except:
                    App.log(0 , "Error : The class {0} does not exist".format(cl))
            else:
                if cl["name"] in files:
                    output_files += self.pick_samples(files , cl["classes"] , count , cl["name"])
                else:
                    output_files += self.pick_samples(files , cl["classes"] , count )
                continue

            idx = np.arange(len(files_cl))
            np.random.shuffle(idx)
            idx = idx[:count]

            files_list = [[cl , index] for index in idx]

            output_files += files_list

        return output_files
