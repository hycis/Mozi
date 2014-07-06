from datetime import datetime
import os
import sys
import logging
import cPickle
import sqlite3

class Log:

    def __init__(self, experiment_name="experiment", description=None,
                save_outputs=False, save_hyperparams=False, save_model=False,
                save_to_database=None, logger=None):

        self.experiment_name = experiment_name
        self.description = description
        self.save_outputs = save_outputs
        self.save_hyperparams = save_hyperparams
        self.save_model = save_model
        self.save_to_database = save_to_database

        dt = datetime.now()
        dt = dt.strftime('%Y%m%d_%H%M_%S%f')

        self.exp_id = experiment_name + '_' + dt

        if save_outputs or save_hyperparams or save_model:
            save_dir = os.environ['PYNET_SAVE_PATH'] + '/log'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)

            self.exp_dir = save_dir + '/' + self.exp_id
            if not os.path.exists(self.exp_dir):
                os.mkdir(self.exp_dir)

        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

        if save_outputs:
            ch = logging.FileHandler(filename=self.exp_dir+'/outputs.log')
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        self.logger.info('exp_id: ' + self.exp_id)

        if description is not None:
            self.logger.info('Description: ' + self.description)

        if save_to_database:
            self.first_time_record = True

    def info(self, msg):
        self.logger.info(msg)

    def print_records(self):

        for key in self.save_to_database['records']:
            self.info(key + ': ' + `self.save_to_database['records'][key]`)

    def _log_outputs(self, outputs):
        dt = datetime.now()
        dt = dt.strftime('%Y-%m-%d %H:%M')
        self.logger.info('Time: ' + dt)

        for (name, val) in outputs:
            self.logger.info(name + ': ' + str(val))

        if self.save_outputs:
            self.logger.info('[ outputs saved to: %s ]\n' %self.exp_id)

        print('\n')

    def _save_model(self, model):
        with open(self.exp_dir+'/model.pkl', 'wb') as pkl_file:
            cPickle.dump(model, pkl_file)

    def _save_hyperparams(self, learning_rule):
        with open(self.exp_dir+'/hyperparams.pkl', 'wb') as pkl_file:
            cPickle.dump(learning_rule, pkl_file)

    def _save_to_database(self, epoch, train_error, valid_error, test_error):
        conn = sqlite3.connect(os.environ['PYNET_DATABASE_PATH'] + '/' + self.save_to_database['name'])
        cur = conn.cursor()

        query = 'CREATE TABLE IF NOT EXISTS ' + self.experiment_name + '(exp_id TEXT PRIMARY KEY NOT NULL,'

        for k,v in self.save_to_database['records'].items():
            if type(v) is str:
                query += k + ' TEXT,'
            elif type(v) is int:
                query += k + ' INT,'
            else:
                query += k + ' REAL,'

        query += 'epoch INT, train_error REAL, valid_error REAL, test_error REAL);'

        cur.execute(query)


        if self.first_time_record:
            try:
                query = 'INSERT INTO ' + self.experiment_name + ' VALUES('
                ls = [self.exp_id]
                for k, v in self.save_to_database['records'].items():
                    query += '?,'
                    ls.append(v)
                query += '?,?,?,?,?);'
                ls.extend([epoch, train_error, valid_error, test_error])
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
                        'train_error = ?,' +
                        'valid_error = ?,' +
                        'test_error = ?' +
                        "WHERE exp_id='%s'"%self.exp_id,
                        [epoch,
                        train_error,
                        valid_error,
                        test_error])

        conn.commit()
        conn.close()
