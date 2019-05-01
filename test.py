import json
import torch
from sqlnet.utils import *
from sqlnet.model.seq2sql import Seq2SQL
from sqlnet.model.sqlnet import SQLNet
import numpy as np
import datetime

import argparse

def gen_sql_data(question):
    result = {} 
    question_tok = question.split(" ")
    result['question'] = question
    result['query_tok'] = []
    result['query_tok_space'] = []
    result['sql'] = {
	  "agg": 0,
	  "sel": 0,
	  "conds": []
	}
    result['table_id'] = '1-1529260-2'
    result['question_tok'] = question_tok
    result['phase'] = 1
    result['query'] = ''
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--toy', action='store_true', 
            help='If set, use small data; used for fast debugging.')
    parser.add_argument('--ca', action='store_true',
            help='Use conditional attention.')
    parser.add_argument('--dataset', type=int, default=0,
            help='0: original dataset, 1: re-split dataset')
    parser.add_argument('--rl', action='store_true',
            help='Use RL for Seq2SQL.')
    parser.add_argument('--baseline', action='store_true', 
            help='If set, then test Seq2SQL model; default is SQLNet model.')
    parser.add_argument('--train_emb', action='store_true',
            help='Use trained word embedding for SQLNet.')
    parser.add_argument('--demo', action='store_true',
            help='demonstrate some handwritten queries')
    args = parser.parse_args()

    N_word=300
    B_word=42
    if args.toy:
        USE_SMALL=True
        GPU=False
        BATCH_SIZE=15
    else:
        USE_SMALL=False
        GPU=False
        BATCH_SIZE=64
    TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)

    sql_data, table_data, val_sql_data, val_table_data, \
            test_sql_data, test_table_data, \
            TRAIN_DB, DEV_DB, TEST_DB = load_dataset(
                    args.dataset, use_small=USE_SMALL)

    word_emb = load_word_emb('glove/glove.%dB.%dd.txt'%(B_word,N_word), \
        load_used=True, use_small=USE_SMALL) # load_used can speed up loading

    if args.baseline:
        model = Seq2SQL(word_emb, N_word=N_word, gpu=GPU, trainable_emb = True)
    else:
        model = SQLNet(word_emb, N_word=N_word, use_ca=args.ca, gpu=GPU,
                trainable_emb = True)

    if args.train_emb:
        agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_name(args)
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m))
        print "Loading from %s"%agg_e
        model.agg_embed_layer.load_state_dict(torch.load(agg_e))
        print "Loading from %s"%sel_e
        model.sel_embed_layer.load_state_dict(torch.load(sel_e))
        print "Loading from %s"%cond_e
        model.cond_embed_layer.load_state_dict(torch.load(cond_e))
    else:
        agg_m, sel_m, cond_m = best_model_name(args)
        if args.demo:
            agg_m = 'model_best_noca/epoch4.agg_model'
            cond_m = 'model_best_noca/epoch47.cond_model'
            sel_m = 'model_best_noca/epoch40.sel_model'
        print "Loading from %s"%agg_m
        model.agg_pred.load_state_dict(torch.load(agg_m, map_location='cpu'))
        print "Loading from %s"%sel_m
        model.sel_pred.load_state_dict(torch.load(sel_m, map_location='cpu'))
        print "Loading from %s"%cond_m
        model.cond_pred.load_state_dict(torch.load(cond_m, map_location='cpu'))
        if args.demo:
            
            with open('question.json') as f:
                sql_data = json.load(f)
                sql_data = [sql_data, sql_data]
                print("sql_data---------- ", sql_data)
            with open('table.json') as f:
                table_data = json.load(f)
                print(table_data)
                table_data = {sql_data[0]['table_id']: table_data}
            sql_data = gen_sql_data("How many dollars is the purse when the margin of victory is 8 strokes")
            sql_data = [sql_data, sql_data]
            print("Predicted Query:", pred_query_string(model, sql_data, table_data, DEV_DB))
            # Process user input question
            tid = list(table_data.keys())[0]
            while(True):
                print("")
                print("-------Table schema---------------")
                print(json.dumps(table_data[tid]['header']))
                print("-------Sample table content-------")
                print(json.dumps(table_data[tid]['rows'][0]))
                print(json.dumps(table_data[tid]['rows'][1]))
                text = raw_input("Ask a Question>")
                sql_data = gen_sql_data(text)
                sql_data = [sql_data, sql_data]
                print("Predicted Query:", pred_query_string(model, sql_data, table_data, DEV_DB))
        else:
            print "Dev_DB %s"%DEV_DB
            # print "Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
			# 		model, BATCH_SIZE, sql_data, table_data, TEST_ENTRY)
            print "Dev execution acc: %s"%epoch_exec_acc( \
                model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)
			# print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
			# 		model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY)
			# print "Test execution acc: %s"%epoch_exec_acc(
			# 		model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB)


