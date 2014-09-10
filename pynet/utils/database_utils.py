import sqlite3


def display_database(path, db):

    conn = sqlite3.connect(path)
    cur = conn.cursor()

    db = cur.execute("SELECT * FROM %s" %db)

    for row in db:
        for e in row:
            print e
        print '\n'

    conn.close()
