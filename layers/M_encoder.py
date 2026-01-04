import torch
import torch.nn as nn
from layers.Embed import DataEmbedding

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

        
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
class Mul_Block(nn.Module):
    def __init__(self, configs, freature_flag):
        super().__init__()
        self.seq_len = configs.seq_len
        self.d_model = configs.oenc_dim
        self.enc_in = configs.enc_in
        self.freq = configs.freq
        self.feature_flag = freature_flag
        self.data_embedding = DataEmbedding(self.enc_in-1, self.d_model, embed_type='fixed', freq=self.freq, dropout=0.1)
        self.LSTM = nn.LSTM(input_size=self.d_model,hidden_size=self.d_model,num_layers=3,batch_first=True)
        self.GRU = nn.GRU(input_size=self.d_model,hidden_size=self.d_model,num_layers=3,batch_first=True)
        self.TRANSFORMER = nn.TransformerEncoderLayer(
                        d_model=self.d_model,
                        nhead=8,
                        batch_first=True)
    def forward(self, x_enc, x_mark_enc):
        dec_out = self.data_embedding(x_enc, x_mark_enc)
        if self.feature_flag == 'LSTM':
            dec_out, (h_n, c_n) = self.LSTM(dec_out)
        elif self.feature_flag == 'GRU':
            dec_out, h_n = self.GRU(dec_out)
        elif self.feature_flag == 'TRANSFORMER':
            dec_out = self.TRANSFORMER(dec_out)
        return dec_out


    
# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument('--d_model', type=int, default=128,
#                     help='input sequence length')
# parser.add_argument('--enc_in', type=int, default=6,
#                     help='prediction sequence length')
# parser.add_argument('--freq', type=str, default='h',
#                     help='input sequence length')
# parser.add_argument('--feature_flag', type=str, default='LSTM',
#                     help='input sequence length')
# parser.add_argument('--seq_len', type=int, default=96,
#                     help='input sequence length')
# args = parser.parse_args()
# if __name__ == '__main__':
#     xenc = torch.randn(1, 96, 6)
#     xmark = torch.zeros(1, 96, 4)
#     model = Mul_Block(args)
#     out = model(xenc, xmark)
#     pass
