import pymysql.cursors
import pdb
import sqlite3

database_type = "mysql"

#MYSQL config
user = 'beda'
password = 'hardcore'
database = 'core'
host = '127.0.0.1'
port = 3306


#SQLite config
database_file = "../data/database.db"


class Database:
    def __init__(self):
        if database_type == "mysql":
            self.connection = pymysql.connect(user = user, password = password, host = host, port = port, database = database, charset = 'utf8')
            self.cursor = self.connection.cursor()
            self.substitution_string = "%s"
        
        elif database_type == "sqlite":
            self.connection = sqlite3.connect(database_file)
            self.cursor = self.connection.cursor()
            self.substitution_string = "?"

        assert self.connection != None


    def execute(self, command, values = None):
        self.cursor.execute(command, values)
        self.commit()


    def execute_many(self, command, values = None):
        self.cursor.executemany(command, values)
        self.commit()
    
    
    def commit(self):
        self.connection.commit()


    def close(self):
        self.cursor.close()
        self.connection.close()


    def __del__(self):
        self.commit()
        self.close()


    def get_all_rows(self, command, args = None):
        if args == None:
            self.cursor.execute(command)
        else:
            self.cursor.execute(command, args)
        
        return [row for row in self.cursor]
    

    def get_all_rows_single_element(self, command, args = None):
        if args == None:
            self.cursor.execute(command)
        else:
            self.cursor.execute(command, args)
        
        return [row for (row,) in self.cursor]
    
    
    def insert(self, table, values):
        if len(values) == 0:
            return
        
        command = "REPLACE INTO " + table + " VALUES (" + ", ".join(len(values) * [self.substitution_string]) + ")"
        self.execute(command, values)
    
    
    def insert_many(self, table, values):
        if len(values) == 0:
            return
        
        command = "REPLACE INTO " + table + " VALUES (" + ", ".join(len(values[0]) * [self.substitution_string]) + ")"
        self.execute_many(command, values)


    
