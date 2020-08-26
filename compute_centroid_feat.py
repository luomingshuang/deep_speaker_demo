import os
import random
import time
import torch
import glob
import numpy as np
from torch.autograd import Variable

import torch
import random
import numpy as np

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity

# Reproducible results.
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

# Load the checkpoint.
model.m.load_weights('weights/ResCNN_triplet_training_checkpoint_265.h5', by_name=True)
data = 'media'

def get_centroid(embeddings, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings):
        if utterance_id <= (utterance_num-1):
            #print(utterance.shape)
            centroid = centroid + utterance 
        else: break
    centroid = centroid/utterance_num
    return centroid

#speakers = ['cl', 'fuli', 'gongwenhua', 'liuyuguang', 'lms', 'lsq', 'lxx', 'lzh', 'shanke', 'wry', 'zhangshuai163', 'zhuting', 'zlb', 'zq', 'yuyaqi']
speakers = ['fuli', 'lzh', 'cl', 'lsq', 'lxx', 'wry', 'lms', 'zlb', 'zq']
enroll_wav_path = 'data_eng'
dict_spkid_embeddings = {}
enroll_nums = 5
total_wavs = glob.glob(os.path.join(data, '*', '*.wav'))
print('total wavs: ', len(total_wavs))
for speaker in speakers:
    #print(speaker)
    speaker_wavs = glob.glob(os.path.join(enroll_wav_path, speaker, '*.wav'))
    length = len(speaker_wavs)
    speaker_embeddings = np.zeros((length, 512), dtype=float)
    print(speaker, length)
    #speaker_embeddings = []
    for i in range(length):
    #for wav in speaker_wavs:
        mfcc = sample_from_mfcc(read_mfcc(speaker_wavs[i], SAMPLE_RATE), NUM_FRAMES)
        predict_feat = model.m.predict(np.expand_dims(mfcc, axis=0))
        #speaker_embeddings.append(predict_feat)
        speaker_embeddings[i] = predict_feat    
    dict_spkid_embeddings[speaker] = speaker_embeddings
    # num_utterances = len(speaker_wavs)
    # enroll_centroid_embeddings = get_centroid(speaker_embeddings, num_utterances)
    # dict_spkid_embeddings[speaker] = enroll_centroid_embeddings
    # print(speaker, enroll_centroid_embeddings.shape)

np.save('enroll_9.npy', dict_spkid_embeddings)