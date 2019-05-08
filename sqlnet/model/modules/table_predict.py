import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class TablePredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca, table_ca=True, dot_out=True):
        super(TablePredictor, self).__init__()
        self.table_ca = table_ca
        self.dot_out = dot_out
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
            self.table_att = nn.Linear(N_h, 1)
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.table_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 2))
        self.sel_out_simple = nn.Sequential(nn.Tanh(), nn.Linear(N_h*2, 2))
        self.softmax = nn.Softmax()
        self.N_h = N_h

    def nice_print(s, x):
        print(s, x.shape, x)

    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)
        max_col_name_len = max(col_name_len)

        # print("col_inp_var", col_inp_var.shape)
        # print("col_name_len", col_name_len)
        # print("col_len", col_len)
        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)
        # table_att_val = self.table_att(e_col).squeeze()
        # for idx, num in enumerate(col_len):
        #     if num < max_col_name_len:
        #         table_att_val[idx, num:] = -100
        # table_att = self.softmax(table_att_val)
        # table_K_sel = (e_col * table_att.unsqueeze(2).expand_as(e_col)).sum(1)
        # table_K_sel_expand=table_K_sel.unsqueeze(1)
        # table_out_K_out = self.table_out_K(table_K_sel_expand)
        # print("table_att_val", table_att_val.shape)
        # print("table_att", table_att.shape)
        # print("table_K_sel", table_K_sel.shape)
        # print("table_K_sel_expand", table_K_sel_expand.shape)
        # print("table_out_K_out", table_out_K_out.shape)

        # print("x_emb_var", x_emb_var.shape)
        # print("x_len", x_len)
        # print("col_inp_var", col_inp_var.shape)
        # print("col_name_len", col_name_len)
        # print("col_len", col_len)
        # print("col_num", col_num)
        # print("e_col", e_col.shape)
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
        # print("N_h", self.N_h)
        # print("h_enc", h_enc.shape)
        # print("att_val", att_val.shape)
        # print("K_sel", K_sel.shape)
        # print("K_sel_expand", K_sel_expand.shape)
        # sel_out_K_out = self.sel_out_K(K_sel_expand)
        # ---- new
        # print("K_sel", K_sel.shape)
        sel_out_K_out = self.sel_out_K(K_sel)
        # print("sel_out_K_out", sel_out_K_out.shape)
        e_sum = torch.sum(e_col, dim=1)
        # print("e_sum", e_sum.shape)
        if self.dot_out:
            sel_score = torch.bmm(sel_out_K_out.view(B, 1, self.N_h), e_sum.view(B, self.N_h, 1))
            sel_score = torch.cat((sel_score, -1*sel_score), 1).squeeze()
        else: 
            cat_out = torch.cat((sel_out_K_out, e_sum), 1)
            # print("cat_out", cat_out.shape)
            sel_score = self.sel_out_simple(cat_out)
            # print("sel_score", sel_score.shape)
        # ---- end new
        # sel_out_col_out = self.sel_out_col(e_col)
        # print("sel_out_col_out", sel_out_col_out.shape)
        # sel_out_plus = sel_out_K_out + table_out_K_out
        # print("sel_out_K_out", sel_out_K_out.shape)
        # print("sel_out_plus", sel_out_plus.shape)
        # sel_score = self.sel_out( sel_out_plus ).squeeze()
        # print("sel_score", sel_score.shape, sel_score)
        # max_col_num = max(col_num)
        # for idx, num in enumerate(col_num):
        #     if num < max_col_num:
        #         sel_score[idx, num:] = -100
        # print("sel_score shape", sel_score.shape)
        # sel_score = self.softmax(sel_score)
        # print("sel_score_sm shape", sel_score.shape)
        # print("final sel_score", sel_score)
        return sel_score
