import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn, optim
from torchvision import transforms, datasets
import torch.nn.functional as F

class Encoder_Decoder(nn.Module):
    def __init__(self, encoder,decoder,sliding_window_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sliding_window_size = sliding_window_size
        
    def forward(self,chroma,beat_note,last_output,cau_dict,start_t,end_t,beat_interval):
        t = start_t
        cau_gen=[]
        this_cau_gen = 42 #'SOD'
        output_concat = []
        # init hidden layer for decoder
        hidden = decoder.init_hidden()
        while this_cau_gen!= 43 and t + 1000 <= end_t:
            # retrive musical features
            chroma_interval = get_chroma_interval(chroma,t,chroma_len=200)
            beat_note_interval = get_beat_note_interval(beat_note,t,beat_note_len=2000)
            chroma_encoded,beat_note_encoded = self.encoder(chroma_interval.float(), beat_note_interval.float())
            acoustic = torch.from_numpy(np.concatenate([chroma_encoded.detach().numpy(),beat_note_encoded.detach().numpy()],axis=1))
            
            # decoder
            output,hidden = decoder(last_output, hidden, acoustic)
            cau_id = np.argmax(np.exp(output.detach().numpy()))
            this_cau_gen = cau_id
            
            # update t, output and save generated CAU sequence
            cau_gen.append(this_cau_gen)
            t += get_beat_of_cau(cau_file,cau_idx_to_name,this_cau_gen,True) * beat_interval
            last_output = torch.from_numpy(np.array(int(this_cau_gen)))
            output_concat.append(np.exp(output.detach().numpy()))
        return cau_gen,output_concat