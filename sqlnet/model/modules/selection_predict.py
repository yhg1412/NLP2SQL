import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        if use_ca:
            print "Using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, N_h)
        else:
            print "Not using column attention on selection predicting"
            self.sel_att = nn.Linear(N_h, 1)
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()
        self.N_h = N_h

    def nice_print(s, x):
        print(s, x.shape, x)

    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)
        print("x_emb_var", x_emb_var.shape)
        print("x_len", x_len)
        print("col_inp_var", col_inp_var.shape)
        print("col_name_len", col_name_len)
        print("col_len", col_len)
        print("col_num", col_num)
        print("e_col", e_col.shape)
        if self.use_ca:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            att = self.softmax(att_val.view((-1, max_x_len))).view(
                    B, -1, max_x_len)
            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        else:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = self.sel_att(h_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val)
            K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            K_sel_expand=K_sel.unsqueeze(1)
        print("N_h", self.N_h)
        print("h_enc", h_enc.shape)
        print("att_val", att_val.shape)
        print("K_sel", K_sel.shape)
        print("K_sel_expand", K_sel_expand.shape)
        sel_out_K_out = self.sel_out_K(K_sel_expand)
        sel_out_col_out = self.sel_out_col(e_col)
        sel_out_plus = sel_out_K_out + sel_out_col_out
        print("sel_out_K_out", sel_out_K_out.shape)
        print("sel_out_col_out", sel_out_col_out.shape)
        print("sel_out_plus", sel_out_plus.shape)
        sel_score = self.sel_out( sel_out_plus ).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100
        print("sel_score shape", sel_score.shape)
        return sel_score
