import argparse
import sqlite3
from operator import itemgetter
import os
import sys

parser = argparse.ArgumentParser(description='remove poor models')
parser.add_argument('--database', metavar='PATH', help='path to the database')
parser.add_argument('--tables', metavar='list of tables', default=False, type=str, nargs='+', help='tables to be executed on')
parser.add_argument('--base_on', metavar='Column', help='remove based on the column values')
parser.add_argument('--descending', action='store_true', help='rank by with smallest value on top' )
parser.add_argument('--keep_topk', metavar='NUMBER', type=int, help='keep the topk models based on the column specified')
parser.add_argument('--model_root_dir', metavar='PATH', help='root directory for the saved models')

args = parser.parse_args()
conn = sqlite3.connect(args.database)
conn.text_factory = str
cur = conn.cursor()

if not args.tables:
    cmd = cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tbls = [t for t, in cmd.fetchall()]
else:
    tbls = args.tables

print '..Tables', tbls
for tbl in tbls:
    try:
        print '..Table', tbl
        cmd = cur.execute("SELECT exp_id, {} FROM '{}';".format(args.base_on, tbl))
        vals = cmd.fetchall()
        vals = sorted(vals, key=itemgetter(1), reverse=args.descending)
        print vals
    
        topk = args.keep_topk if args.keep_topk < len(vals) else len(vals)
        print '..topk', topk
        for i in xrange(topk, len(vals)):
            exp_id = vals[i][0]
            print '..removing', vals[i]
            print os.system('rm -r {}/{}'.format(args.model_root_dir, exp_id))
    except:
        print '..Error', sys.exc_info()[0]

conn.close()
