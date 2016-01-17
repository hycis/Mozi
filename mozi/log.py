
from datetime import datetime
import os
import sys
import logging
import cPickle
import sqlite3
import operator
import copy
import numpy as np

import theano
from theano.sandbox.cuda.var import CudaNdarraySharedVariable

floatX = theano.config.floatX

class Log:

    def __init__(self, experiment_name="experiment", description=None,
                save_outputs=False, save_model=False,
                save_epoch_error=False, save_to_database=None, logger=None):

        self.experiment_name = experiment_name
        self.description = description
        self.save_outputs = save_outputs
        self.save_model = save_model
        self.save_epoch_error = save_epoch_error
        self.save_to_database = save_to_database

        dt = datetime.now()
        dt = dt.strftime('%Y%m%d_%H%M_%S%f')

        self.exp_id = experiment_name + '_' + dt

        if save_outputs or save_model:
            save_dir = os.environ['MOZI_SAVE_PATH']
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            self.exp_dir = save_dir + '/' + self.exp_id
            if not os.path.exists(self.exp_dir):
                os.mkdir(self.exp_dir)

        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

        # self.logger.info('exp_id: ' + self.exp_id)

        if save_outputs:
            ch = logging.FileHandler(filename=self.exp_dir+'/outputs.log')
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if save_epoch_error:
            self.epoch_error_path = self.exp_dir+'/epoch_error.csv'
            with open(self.epoch_error_path, 'wb') as epoch_file:
                epoch_file.write('Epoch,Error\n')

        if description is not None:
            self.logger.info('Description: ' + self.description)

        if save_to_database:
            self.first_time_record = True
            if not os.path.exists(os.environ['MOZI_DATABASE_PATH']):
                os.mkdir(os.environ['MOZI_DATABASE_PATH'])

    def info(self, msg):
        self.logger.info(msg)

    def print_records(self):
        sorted_ls = sorted(self.save_to_database['records'].iteritems(),
                             key=operator.itemgetter(0))
        for key, value in sorted_ls:
            self.info(key + ': ' + str(value))

    def _log_outputs(self, outputs):
        dt = datetime.now()
        dt = dt.strftime('%Y-%m-%d %H:%M')
        self.logger.info('Time: ' + dt)

        for (name, val) in outputs:
            self.logger.info(name + ': ' + str(val))

        if self.save_outputs:
            self.logger.info('[ outputs saved to: %s ]\n' %self.exp_id)

    def _save_model(self, model):
        with open(self.exp_dir+'/model.pkl', 'wb') as pkl_file:
            cPickle.dump(model, pkl_file)

    def _save_epoch_error(self, epoch, valid_error):
        with open(self.epoch_error_path, 'ab') as epoch_file:
            epoch_file.write('{},{}\n'.format(epoch, valid_error))

    def _save_to_database(self, epoch, train_cost, valid_cost, best_valid_error):
        conn = sqlite3.connect(os.environ['MOZI_DATABASE_PATH'] + '/' + self.save_to_database['name'])
        cur = conn.cursor()

        if self.first_time_record:
            query = 'CREATE TABLE IF NOT EXISTS ' + self.experiment_name + \
                    '(exp_id TEXT PRIMARY KEY NOT NULL,'

            for k,v in self.save_to_database['records'].items():
                if type(v) is str:
                    query += k + ' TEXT,'
                elif type(v) is int:
                    query += k + ' INT,'
                elif type(v) is float:
                    query += k + ' REAL,'
                else:
                    raise Exception("Error: The input types for records '{}' of {}".format(k, type(v))
                                    + " is not primitive types (str, int, float).")

            query += 'epoch INT, train_cost REAL, valid_cost REAL, best_valid_error REAL);'

            cur.execute(query)

            try:
                query = 'INSERT INTO ' + self.experiment_name + ' VALUES('
                ls = [self.exp_id]
                for k, v in self.save_to_database['records'].items():
                    query += '?,'
                    ls.append(v)
                query += '?,?,?,?,?);'
                ls.extend([epoch, train_cost, valid_cost, best_valid_error])
                cur.execute(query, ls)
                self.first_time_record = False

            except sqlite3.OperationalError as err:
                self.logger.error('sqlite3.OperationalError: ' + err.message)
                self.logger.error('Solution: Change the experiment_name in Log() to a new name, '
                        + 'or drop the same table name from the database. '
                        + 'experiment_name is used as the table name.')
                raise

        else:
            cur.execute('UPDATE ' + self.experiment_name + ' SET ' +
                        'epoch = ?, ' +
                        'train_cost = ?,' +
                        'valid_cost = ?,' +
                        'best_valid_error = ?' +
                        "WHERE exp_id='%s'"%self.exp_id,
                        [epoch,
                        train_cost,
                        valid_cost,
                        best_valid_error])
        conn.commit()
        conn.close()
