import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,
                 num_his = 1,
                 num_one_step_obs = 1,
                 d_model = 64,
                 num_heads = 16,
                 feedforward_dim = 64*4,
                 dropout=0.1,
                 activation='elu',
                 self_attn=False):
        super(Encoder, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_his = num_his
        self.num_one_step_obs = num_one_step_obs
        self.self_attn = self_attn
        self.activation = get_activation(activation)
        self.prop_embed = nn.Linear(num_one_step_obs, d_model)
        
        if num_his > 1 and self_attn:
            self.position_encoding = PositionEncoding(d_model=d_model, len=num_his)
            self.prop_self_attn = MultiHeadAttn(d_model=d_model, num_heads=num_heads, dropout=dropout)
        
        self.cnn_for_height_map = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # output size: (16, H/2, W/2)
            self.activation,
            nn.Conv2d(16, d_model, kernel_size=3, stride=1, padding=1),  # output size: (d_model, H/2, W/2)
            self.activation,
        )
        # self.W_q = nn.Linear(d_model, d_model)
        # self.W_k = nn.Linear(d_model, d_model)
        # self.MHA = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.MHA = MultiHeadAttn(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            self.activation,
            nn.Linear(feedforward_dim, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, obs_history, height_map):
        batch_size = obs_history.shape[0]
        
        assert height_map.shape[0] == batch_size, "Batch size of height_map must match obs_history"
        
        height_feature = self.cnn_for_height_map(height_map)  # (batch_size, d_model, H, W)
        height_feature = height_feature.flatten(2).permute(0, 2, 1)  # (batch_size, H*W, d_model)
        
        prop_feature = self.prop_embed(obs_history.view(batch_size, self.num_his, self.num_one_step_obs))  # (batch_size, num_his, num_one_step_obs, d_model)
        if self.num_his > 1 and self.self_attn:
            prop_feature = self.position_encoding(prop_feature)  # (batch_size, num_his, d_model)
            prop_feature, _ = self.prop_self_attn(prop_feature, prop_feature, prop_feature)  # (batch_size, num_his, d_model)
        attn_output, _ = self.MHA(prop_feature, height_feature, height_feature) # (batch_size, H*W, d_model)
        
        attn_output = self.layer_norm1(self.dropout1(attn_output) + prop_feature)  # (batch_size, num_his, d_model)
        
        ffn_output = self.ffn(attn_output)  # (batch_size, num_his, d_model)
        out = self.layer_norm2(self.dropout2(ffn_output) + attn_output)  # (batch_size, num_his, d_model)
        
        return out
    
class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttn, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_q, d_k)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)    # (batch_size, num_heads, seq_len_k, d_k)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len_v, d_k)
        
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale.to(Q.device)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len_q, d_k)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model) 
        output = self.fc(output)  # (batch_size, seq_len_q, d_model)
        return output, attn_weights
    
class PositionEncoding(nn.Module):
    def __init__(self, d_model, len):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, len, d_model)
        self.register_buffer('pe', pe) # (1, len, d_model) self.pe是一个持久缓冲区，不是模型参数，不会被优化器更新，但会随着模型一起保存和加载。

    def forward(self, x):
        x = x + self.pe
        return x
    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None