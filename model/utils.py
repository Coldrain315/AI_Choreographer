import numpy as np
import pandas as pd
import json
import torch

def unpack_music_features(feature_dir,genre,idx):
    this_dir = feature_dir + genre + '/' + str(idx) + '.json'
#     print('==> Dance Genre:', genre,', No.',idx)
    with open(this_dir) as file:
        music_feature = json.loads(file.read())
        chroma_vector = np.array(music_feature['chroma'])
        beat_vector = np.array(music_feature['beat'])
        onset_vector = np.array(music_feature['onset'])
    return chroma_vector,beat_vector,onset_vector

def load_data():
    genre = {'C':9,'R':10,'T':9,'W':34}
    feature_dir = './music_features/'
    for dancetype,_ in genre.items():
        for num in range(1,genre[dancetype]+1):
            chroma_vector,beat_vector,onset_vector = unpack_music_features(feature_dir,dancetype,num)
            

def get_song_idx(genre):
    song_list = []
    song_idx = []
    song_dir = './choreography/' + genre + '.json'
    print('==> Loading Song name of Genre', genre,'from',song_dir)
    with open(song_dir) as file:
        songs = json.loads(file.read())
        for i in range(len(songs)):
            song_list.append(songs[i]['name'])
            song_idx.append(i)
    return song_list,song_idx

def get_song_info(cau_file,cau_idx_to_name,genre,idx):
    song_dir = './choreography/' + genre +'.json'
    with open(song_dir) as file:
        all_songs = json.loads(file.read())
        this_song = all_songs[idx]
        dance_name = this_song['name']
        dance_id = this_song['dance_id']
        start_pos = this_song['start_pos']
        end_pos = this_song['end_pos']
        gt_cau = this_song['movements']
        beat_len = 0
        for cau in gt_cau:
            cau_beat_len = get_beat_of_cau(cau_file,cau_idx_to_name,cau,False)
            beat_len += cau_beat_len
        interval = ((end_pos-start_pos)*4/beat_len)
    return dance_name, dance_id, gt_cau, start_pos, end_pos,interval

def load_CAU_dict(genre):
    # CAU loader C: 41, R:42, T:45, W:32
    cau_list = []
    cau_dir = './choreography/' + genre + '.json'
    print('==> Retrieving CAU list of genre', genre,'from',cau_dir)
    with open(cau_dir) as file:
        cau = json.loads(file.read())
        for i in range(len(cau)):
            for move in cau[i]['movements']:
                cau_list.append(move)
    cau_list = np.unique(cau_list)
    cau_mapping = dict(enumerate(cau_list.flatten(), 1))
    cau_mapping[len(cau_list)] = 'NIL'
    cau_mapping[len(cau_list)+1] = 'SOD'
    cau_mapping[len(cau_list)+2] = 'EOD'
    cau_mapping[len(cau_list)+3] = 'HOLD'
    cau_idx_to_name = cau_mapping
    cau_name_to_idx = {v: k for k, v in cau_idx_to_name.items()}
    return cau_idx_to_name,cau_name_to_idx

def get_beat_of_cau(cau_file,cau_idx_to_name,cau_name,idx=False):
    if cau_name in ('HOLD','SOD','EOD','NIL',41,42,43,44):
        beat = 4
    elif idx== True:
        beat = cau_file.loc[cau_file['movement_tag'] == cau_idx_to_name[cau_name+1], 'beats'].values[0]
    else:
        print(cau_name)
        beat =  cau_file.loc[cau_file['movement_tag'] == cau_name, 'beats'].values[0]
    return beat

def align_cau_gen(song_len, cau_seq, prob_seq):
    
    aligned_cau_gen = []
    aligned_prob_gen = []
    
    beat_idx = 0
    for i in range(len(cau_seq)):
        cau = cau_seq[i]
        prob = prob_seq[i]
        beats_len = get_beat_of_cau(cau_file,cau,True) * interval
        for j in range(int(beat_idx),int(beat_idx+beats_len)):
            if j < song_len:
                aligned_cau_gen.append(cau)
                aligned_prob_gen.append(prob)
            else:
                break
        beat_idx+=beats_len
    for blanks in range(int(beat_idx),song_len):
        aligned_cau_gen.append(41)
        aligned_prob_gen.append(torch.zeros(45))
    return aligned_cau_gen,aligned_prob_gen

def align_cau_gt(song_len, cau_seq):
    aligned_cau_gt = []    
    beat_idx = 0
    for i in range(len(cau_seq)):
        cau = cau_seq[i]
        beats_len = get_beat_of_cau(cau_file,cau,True) * interval
        for j in range(int(beat_idx),int(beat_idx+beats_len)):
            if j < song_len:
                aligned_cau_gt.append(cau)
            else:
                break
        beat_idx+=beats_len
    for blanks in range(int(beat_idx),song_len):
        aligned_cau_gt.append(41)
    return aligned_cau_gt
