'''
CS7180: Advanced Perception
Name: Sumegha Singhania
Date: 11.11.22
This file pertains to preparing the data for training. 
The audio samples are converted into spectrograms and 
binary labels are associated with them, depending on whether
they are desired or undesired/background noises
'''

import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

def prepare_data():
    # Loading and preparing the data
    def load_wav_files(file):
        # Load encoded file
        contents = tf.io.read_file(file)
        # Decode file (tensors by channels) 
        wav, sample_rate = tf.audio.decode_wav(contents, desired_channels=1)
        # Removes trailing axis
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        # Goes from 44100Hz to 16000hz - amplitude of the audio signal
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    # Path to files
    # Positive data set represents the audio signal we desire to find
    # Negative data set represents all the external noises in the audio files
    POS = os.path.join('data', 'Parsed_Capuchinbird_Clips')
    NEG = os.path.join('data', 'Parsed_Not_Capuchinbird_Clips')

    # Convert to datasets
    pos = tf.data.Dataset.list_files(POS+'\*.wav')
    neg = tf.data.Dataset.list_files(NEG+'\*.wav')

    # adding label 1 to the positive dataset
    desired = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    # adding label 0 to the negative dataset
    external = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    # combining the positive and negative data
    data = desired.concatenate(external)

    # determine average length of the desired sound sample
    lengths = []
    for file in os.listdir(os.path.join('data', 'Parsed_Capuchinbird_Clips')):
        tensor_wave = load_wav_files(os.path.join('data', 'Parsed_Capuchinbird_Clips', file))
        lengths.append(len(tensor_wave))

    # finding the mean, min and max of the sounds to
    # approximate the length we will set for the spectogram
    tf.math.reduce_mean(lengths)
    tf.math.reduce_min(lengths)
    tf.math.reduce_max(lengths)

    # Convert the sound signals to images by converting them into spectrograms
        # Pre-processing function for spectogram
        # The padding value will change based on what we decide the sound sample length to be
        # (4800 here)
    def preprocess_spectogram(file_path, label): 
        wav = load_wav_files(file_path)
        wav = wav[:48000]
        zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram, label

    # Testing out the spectogram
    filepath, label = desired.shuffle(buffer_size=10000).as_numpy_iterator().next()
    spectrogram, label = preprocess_spectogram(filepath, label)
    plt.figure(figsize=(30,20))
    plt.imshow(tf.transpose(spectrogram)[0])
    plt.show()

    # Creating training and testing data
        # creating spectograms for the entire dataset
    data = data.map(preprocess_spectogram)
    data = data.cache()
    data = data.shuffle(buffer_size=1000)
    data = data.batch(16)
    data = data.prefetch(8)

        # Using about 70% of the data as training data
    train = data.take(36)
    test = data.skip(36).take(15)
    return test,train