import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def init_database(filename):
    database = filename

    sql_create_user_table = """ CREATE TABLE IF NOT EXISTS users (
                                    "id"	INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT,
                                    "name"	text NOT NULL,
                                    "password"	text NOT NULL,
                                    "role"	text NOT NULL,
                                    "created_date"	text,
                                    "modified_date"	text
                                ); """

    sql_create_tasks_table = """CREATE TABLE IF NOT EXISTS tasks (
                                    id integer PRIMARY KEY AUTOINCREMENT,
                                    id_number text NOT NULL,
                                    defect_type text NOT NULL,
                                    image_path text,
                                    captured_date text
                                );"""

    # create a database connection
    with create_connection(database) as conn:

        # create tables
        if conn is not None:
            # create users table
            create_table(conn, sql_create_user_table)

            # create app table
            create_table(conn, sql_create_tasks_table)
        else:
            print("Error! cannot create the database connection.")


def delete_user(conn, id):
    """
    Delete a user by user id
    :param conn:  Connection to the SQLite database
    :param id: id of the user
    :return:
    """
    sql = 'DELETE FROM users WHERE id=?'
    cur = conn.cursor()
    cur.execute(sql, (id,))
    conn.commit()

def update_user(conn, user):
    """
    update name, password, role, created_date and modified_date of a user
    :param conn:
    :param user:
    :return: user id
    """
    sql = ''' UPDATE users
              SET name = ? ,
                  password = ? ,
                  role = ?,
                  created_date = ?,
                  modified_date = ?
              WHERE id = ?'''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()

def create_user(conn, user):
    """
    Create a new user
    :param conn:
    :param user:
    :return:
    """

    sql = ''' INSERT INTO users(name, password, role, created_date, modified_date)
              VALUES(?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

def select_user_by_username(conn, username):
    """
    Query user by priority
    :param conn: the Connection object
    :param username:
    :return:
    """
    # cur = conn.cursor()
    rows=conn.fetchone("SELECT * FROM users WHERE name=?", (username,))

    # rows = cur.fetchone()

    # for row in rows:
    #     print(row)
    return rows


class DbContext:
    def __init__(self, db_name):
        self.db_name = db_name
        self.conn = None

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()

    def execute(self, query, params=None):
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        self.conn.commit()
        return cursor

    def fetchall(self, query, params=None):
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def fetchone(self, query, params=None):
        cursor = self.execute(query, params)
        return cursor.fetchone()