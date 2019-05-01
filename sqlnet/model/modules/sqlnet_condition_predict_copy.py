import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from net_utils import run_lstm, col_name_encode

class SQLNetCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca=True, gpu=False):
        super(SQLNetCondPredictor, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.use_ca = use_ca

        self.number_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.number_att = nn.Linear(N_h, 1)
        self.number_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 5))
        self.number_encode_colname = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.number_col_att = nn.Linear(N_h, 1)
        self.number_hid1 = nn.Linear(N_h, 2*N_h)
        self.number_hid2 = nn.Linear(N_h, 2*N_h)

        self.column_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.column_att = nn.Linear(N_h, N_h)
        self.column_encode_colname = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.column_out_K = nn.Linear(N_h, N_h)
        self.column_out_col = nn.Linear(N_h, N_h)
        self.column_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.op_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.op_att = nn.Linear(N_h, N_h)
        self.op_out_K = nn.Linear(N_h, N_h)
        self.op_encode_colname = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.op_out_col = nn.Linear(N_h, N_h)
        self.op_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 3))

        self.str_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.str_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)
        self.str_encode_colname = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        self.str_out_g = nn.Linear(N_h, N_h)
        self.str_out_h = nn.Linear(N_h, N_h)
        self.str_out_col = nn.Linear(N_h, N_h)
        self.str_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()

    def generate_condition_batch(self, tokens):
        dim = len(tokens)
        max_len = max(max([max([len(tok) for tok in token_seq] + [0]) for token_seq in tokens])-1, 1)

        arr = np.zeros((dim, 4, max_len, self.max_tok_num), dtype=np.float32)
        lens = np.zeros((dim, 4))

        for i, token_seq in enumerate(tokens):
            idx = 0
            for idx, token in enumerate(token_seq):
                sel_token = token[:-1]
                lens[i, idx] = len(sel_token)
                for j, t in enumerate(sel_token):
                    arr[i, idx, j, t] = 1
            if idx < 3:
                arr[i, idx+1:, 0, 1] = 1
                lens[i, idx+1:] = 1

        arr_input = Variable(torch.from_numpy(arr))

        return arr_input, lens

    def standardize_value(self, column_vec, lengths, max_length):
    	dim = len(column_vec.shape)
    	for i, num in enumerate(lengths):
    		if num < max_length:
    			if dim == 2:
    				column_vec[i, num:] = -100
    			elif dim == 3:
    				column_vec[i, :, num:] = -100
    			else:
    				column_vec[i, :, :, num:] = -100
    	return column_vec

    def forward(self, x_emb, x_lens, col_input_emb, col_name_lens, col_lens, col_num, op_where, op_conditions, reinforce=False):
    	dim = len(x_lens)
    	max_col_len = max(col_lens)
    	max_x_len = max(x_lens)
		
        '''
        predict number of where conditions
        '''
    	encoded_col_name, _ = col_name_encode(col_input_emb, col_name_lens, col_lens, self.number_encode_colname)
    	col_val = self.number_col_att(encoded_col_name).squeeze()

    	# only maintain relevant column values
    	col_val = self.standardize_value(col_val, col_lens, max_col_len)

    	col_softmax = self.softmax(col_val)
    	K_num_col = (encoded_col_name * col_softmax.unsqueeze(2)).sum(1)

    	# create hidden units and use them to generate encoded tensors for predicting the number
    	hidden1 = self.number_hid1(K_num_col).view(dim, 4, self.N_h//2).transpose(0, 1).contiguous()
    	hidden2 = self.number_hid2(K_num_col).view(dim, 4, self.N_h//2).transpose(0, 1).contiguous()
    	num_encoded, _ = run_lstm(self.number_lstm, x_emb, x_lens, hidden=(hidden1, hidden2))

    	num_val = self.number_att(num_encoded).squeeze()
    	num_val = self.standardize_value(num_val, col_lens, max_col_len)

    	num_softmax = self.softmax(num_val)
    	K_num = (num_encoded * num_softmax.unsqueeze(2).expand_as(num_encoded)).sum(1)
    	number_score = self.number_out(K_num)

    	'''
		predict columns in where clause
    	'''
    	encoded_col_name2, _ = col_name_encode(col_input_emb, col_name_lens, col_lens, self.column_encode_colname)
    	col_encoded, _ = run_lstm(self.column_lstm, x_emb, x_lens)

    	col_val = torch.bmm(encoded_col_name2, self.column_att(col_encoded).transpose(1,2))
    	col_val = self.standardize_value(col_val, x_lens, max_x_len)

    	col_softmax = self.softmax(col_val.view(-1, max_x_len)).view(dim, -1, max_x_len)
    	K_col = (col_encoded.unsqueeze(1) * col_softmax.unsqueeze(3)).sum(2)

    	col_score = self.column_out(self.column_out_K(K_col) + self.column_out_col(encoded_col_name2)).squeeze()
    	col_score = self.standardize_value(col_score, col_lens, max_col_len)

    	'''
		predict operator(s) in the where clause
    	'''
    	if op_conditions:
    		chosen_ops = [[x[0] for x in cond] for cond in op_conditions]
    	else:
    		# if no op_conditions, use the meaningful value we generated earlier (those that are not -100)
    		sorted_nums = np.argmax(number_score.data.cpu().numpy(), axis=1)
    		chosen_ops = [list(np.argsort(-col_score.data.cpu().numpy()[b])[:sorted_nums[b]]) for b in range(len(sorted_nums))]


    	encoded_col_name3, _ = col_name_encode(col_input_emb, col_name_lens, col_lens, self.op_encode_colname)
    	col_emb = []
    	for i in range(dim):
    		emb = torch.stack([encoded_col_name3[i, j] for j in chosen_ops[i]] + [encoded_col_name3[i, 0]] * (4-len(chosen_ops[i])))
    		col_emb.append(emb)
    	col_emb = torch.stack(col_emb)

    	op_encoded, _ = run_lstm(self.op_lstm, x_emb, x_lens)
    	op_val = torch.matmul(self.op_att(op_encoded).unsqueeze(1), col_emb.unsqueeze(3)).squeeze()
    	op_val = self.standardize_value(op_val, x_lens, max_x_len)

    	op_softmax = self.softmax(op_val)
    	K_op = (op_encoded.unsqueeze(1) * op_softmax.unsqueeze(3)).sum(2)

    	op_score = self.op_out(self.op_out_K(K_op) + self.op_out_col(col_emb)).squeeze()

    	'''
		predict values in the where clause
    	'''
    	str_encoded, _ = run_lstm(self.str_lstm, x_emb, x_lens)
    	encoded_col_name4, _ = col_name_encode(col_input_emb, col_name_lens, col_lens, self.str_encode_colname)

    	col_emb = []
    	for i in range(dim):
    		emb = torch.stack([encoded_col_name4[i, j] for j in chosen_ops[i]] + [encoded_col_name4[i, 0]] * (4-len(chosen_ops[i])))
    		col_emb.append(emb)
    	col_emb = torch.stack(col_emb)

    	if op_where:
    		tok_seqs, tok_lens = self.generate_condition_batch(op_where)
    		str_s, _ = self.str_decoder(tok_seqs.view(dim*4, -1, self.max_tok_num))
    		str_s = str_s.contiguous().view(dim, 4, -1, self.N_h)

    		str_encoded_h = str_encoded.unsqueeze(1).unsqueeze(1)
    		str_s_g = str_s.unsqueeze(3)
    		col_emb = col_emb.unsqueeze(2).unsqueeze(2)

    		str_score = self.str_out(self.str_out_h(str_encoded_h) + self.str_out_g(str_s_g) + self.str_out_col(col_emb)).squeeze()
    		str_score = self.standardize_value(str_score, x_lens, max_x_len)

    	else:
            str_encoded_h = str_encoded.unsqueeze(1).unsqueeze(1)
            col_emb = col_emb.unsqueeze(2).unsqueeze(2)
            str_scores = []

            input_matrix = np.zeros((dim*4, 1, self.max_tok_num), dtype=np.float32)
            input_matrix[:, 0, 0] = 1
            input_matrix = Variable(torch.from_numpy(input_matrix))

            str_s, hidden = self.str_decoder(input_matrix)

            for i in range(50):
      	        str_s, hidden = self.str_decoder(input_matrix, hidden)
      	        str_s = str_s.view(dim, 4, 1, self.N_h)
                str_s_g = str_s.unsqueeze(3)

               	str_score = self.str_out(self.str_out_h(str_encoded_h) + self.str_out_g(str_s_g) + self.str_out_col(col_emb)).squeeze()
                str_score = self.standardize_value(str_score, x_lens, max_x_len)

               	_, ans_tok = str_score.view(dim*4, max_x_len).max(1)
               	ans_tok = ans_tok.data

               	input_matrix = Variable(torch.zeros(dim*4, self.max_tok_num).scatter_(1, ans_tok.unsqueeze(1), 1)).unsqueeze(1)

                str_scores.append(str_score)

            str_score = torch.stack(str_scores, 2)
            str_score = self.standardize_value(str_score, x_lens, max_x_len)
        return (number_score, col_score, op_score, str_score)




