import json
from lib.dbengine import DBEngine
import re
import numpy as np
import json
import random
#from nltk.tokenize import StanfordTokenizer

def load_data(sql_paths, table_paths, use_small=False):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print "Loading data from %s"%SQL_PATH
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if use_small and idx >= 1000:
                    break
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print "Loading data from %s"%TABLE_PATH
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data

    return sql_data, table_data

def load_dataset(dataset_id, use_small=False):
    if dataset_id == 0:
        print "Loading from original dataset"
        sql_data, table_data = load_data('data/train_tok.jsonl',
                'data/train_tok.tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
                'data/dev_tok.tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data/test_tok.jsonl',
                'data/test_tok.tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data/train.db'
        DEV_DB = 'data/dev.db'
        TEST_DB = 'data/test.db'
    else:
        print "Loading from re-split dataset"
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl', use_small=use_small)
        TRAIN_DB = 'data_resplit/table.db'
        DEV_DB = 'data_resplit/table.db'
        TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def best_model_name(args, for_load=False):
    new_data = 'new' if args.dataset > 0 else 'old'
    mode = 'seq2sql' if args.baseline else 'sqlnet'
    if for_load:
        use_emb = use_rl = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''
        use_rl = 'rl_' if args.rl else ''
    use_ca = '_ca' if args.ca else ''

    agg_model_name = 'saved_model/%s_%s%s%s.agg_model'%(new_data,
            mode, use_emb, use_ca)
    sel_model_name = 'saved_model/%s_%s%s%s.sel_model'%(new_data,
            mode, use_emb, use_ca)
    cond_model_name = 'saved_model/%s_%s%s%s.cond_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)

    if not for_load and args.train_emb:
        agg_embed_name = 'saved_model/%s_%s%s%s.agg_embed'%(new_data,
                mode, use_emb, use_ca)
        sel_embed_name = 'saved_model/%s_%s%s%s.sel_embed'%(new_data,
                mode, use_emb, use_ca)
        cond_embed_name = 'saved_model/%s_%s%s%s.cond_embed'%(new_data,
                mode, use_emb, use_ca)

        return agg_model_name, sel_model_name, cond_model_name,\
                agg_embed_name, sel_embed_name, cond_embed_name
    else:
        return agg_model_name, sel_model_name, cond_model_name


def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq

def to_batch_seq_table(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    q_seq = []
    col_seq = []
    col_num = []
    ans_seq = []
    query_seq = []
    gt_cond_seq = []
    vis_seq = []
    label_sql = []
    table_ids = table_data.keys()
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
        # question match table
        label_sql.append(1)
        
        table_id = random.choice(table_ids)
        q_seq.append(sql['question_tok'])
        col_seq.append(table_data[table_id]['header_tok'])
        col_num.append(len(table_data[table_id]['header']))
        ans_seq.append((sql['sql']['agg'],
            sql['sql']['sel'], 
            len(sql['sql']['conds']),
            tuple(x[0] for x in sql['sql']['conds']),
            tuple(x[1] for x in sql['sql']['conds'])))
        query_seq.append(sql['query_tok'])
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[table_id]['header'], sql['query']))
        # question doesn't match table
        label_sql.append(0)
        
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq, label_sql
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, label_sql

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids

def epoch_train_table(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0

    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, label_seq = \
                to_batch_seq_table(sql_data, table_data, perm, st, ed)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq, label_seq= label_seq)
        # print("Loss", loss.data.cpu().numpy())
        cum_loss += loss.data.cpu().numpy() * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)

def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        # print(loss.data.cpu().numpy())
        cum_loss += loss.data.cpu().numpy() * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)

def epoch_acc_table(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    tot_num = 0.0
    print_every = 0
    # print("sql_data length", len(sql_data))
    # print("sql_data 0:2", sql_data[0:2])

    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data, label_seq = \
            to_batch_seq_table(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        # if print_every % 10 == 0:
            # print("sql_data", json.dumps(sql_data[st]))
            # print("table_data", json.dumps(table_data[sql_data[perm[st]]['table_id']]))
            # print("Score", score.shape, score)
            # print("label_seq", label_seq)
        
        # print_every+=1

        tot_err = model.check_table_acc(score, label_seq)

        # one_acc_num += (ed-st-one_err)
        tot_acc_num += tot_err
        tot_num += len(q_seq)

        st = ed
    return (tot_num-tot_acc_num) / tot_num, 0 / len(sql_data)

def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)

def pred_query_string(model, sql_data, table_data, db_path):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
        to_batch_seq(sql_data, table_data, perm, 0, 2, ret_vis_data=True)
    score = model.forward(q_seq, col_seq, col_num, (True, True, True))
    raw_q_seq = [x[0] for x in raw_data]
    raw_col_seq = [x[1] for x in raw_data]
    pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))
    tid = list(table_data.keys())[0]
    sql_pred = pred_queries[0]
    result_query = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'], cols=table_data[tid]['header'])
    return result_query

def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    print_every = 0
    # print("sql_data length", len(sql_data))
    # print("sql_data 0:2", sql_data[0:2])

    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        # if print_every % 10 == 0:
        #     print("sql_data", json.dumps(sql_data[st]))
        #     print("table_data", json.dumps(table_data[sql_data[perm[st]]['table_id']]))
        #     print("Agg Score", score[0].shape, col_num)
        #     print(score[0])
        #     print("Sel Score", score[1].shape, np.max(col_num))
        #     print(score[1])
        #     cond_num_score,cond_col_score,cond_op_score,cond_str_score =\
        #                 [x.data.cpu().numpy() for x in score[2]]
        #     print("Cond Score")
        #     print(cond_num_score.shape)
        #     print(cond_col_score.shape)
        #     print(cond_op_score.shape)
        #     print(cond_str_score.shape)
        # print_every+=1
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        # if print_every % 10 == 0:
        #     print(pred_queries)
        one_err, tot_err = model.check_acc(raw_data,
                pred_queries, query_gt, pred_entry)
        
        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def epoch_reinforce_train(model, optimizer, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.train()
    perm = np.random.permutation(len(sql_data))
    cum_reward = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data =\
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, (True, True, True),
                reinforce=True, gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq, raw_q_seq,
                raw_col_seq, (True, True, True), reinforce=True)

        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        rewards = []
        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(1)

        cum_reward += (sum(rewards))
        optimizer.zero_grad()
        model.reinforce_backward(score, rewards)
        optimizer.step()

        st = ed

    return cum_reward / len(sql_data)


def load_word_emb(file_name, load_used=False, use_small=False):
    if not load_used:
        print ('Loading word embedding from %s'%file_name)
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (use_small and idx >= 5000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(map(lambda x:float(x), info[1:]))
        return ret
    else:
        print ('Load used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val

