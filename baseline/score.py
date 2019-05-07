import json
import re
import numpy as np
import pickle
import argparse
import records
from babel.numbers import parse_decimal, NumberFormatError
from simple_baseline import SimpleModel

schema_re = re.compile(r'\((.+)\)')
num_re = re.compile(r'[-+]?\d*\.\d+|\d+')

agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

class DBEngine:

    def __init__(self, fdb):
        #fdb = 'data/test.db'
        self.db = records.Database('sqlite:///{}'.format(fdb))

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name', name=table_id).all()[0].sql.replace('\n','')
        schema_str = schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and (isinstance(val, str) or isinstance(val, unicode)):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError as e:
                    val = float(num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        #print query
        out = self.db.query(query, **where_map)
        return [o.result for o in out]


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


def load_dataset(use_small=False):
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

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB


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

def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids


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


def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data,
                pred_queries, query_gt, pred_entry)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)

def baseline_acc(sql_queries):
    acc = 0
    for sql in sql_queries:
        if "maximum" in sql['question_tok']:
            acc += 'max' in sql['query_tok']
        elif 'minimum' in sql['question_tok']:
            acc += 'min' in sql['query_tok']
        elif 'how many' in ' '.join(sql['question_tok']):
            acc += 'count' in sql['query_tok']
        elif 'total' in sql['question_tok']:
            acc += 'sum' in sql['query_tok']
        elif 'mean' in sql['question_tok'] or 'average' in sql['question_tok']:
            acc += 'avg' in sql['query_tok']
        else:
            acc += all(['max' not in sql['query_tok'], 'min' not in sql['query_tok'],
                                'count' not in sql['query_tok'], 'sum' not in sql['query_tok'],
                                'avg' not in sql['query_tok']])
    return acc*1.0 / len(sql_queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model file location')
    args = parser.parse_args()

    USE_SMALL = True
    BATCH_SIZE = 15
    TEST_ENTRY = (True, True, True)

    sql_data, table_data, val_sql_data, val_table_data, \
        test_sql_data, test_table_data, \
        TRAIN_DB, DEV_DB, TEST_DB = load_dataset(use_small=USE_SMALL)
   
    model = pickle.load(open(args.model, 'rb'))

    print "Dev acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY)
    print "Dev execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB)
    print "Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY)
    print "Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB)

    train_agg_acc = baseline_acc(sql_data)
    dev_agg_acc = baseline_acc(val_sql_data)

    print "Better baseline: Train AGG accuracy: {0}".format(train_agg_acc)
    print "Better baseline: Dev AGG accuracy: {0}".format(dev_agg_acc)