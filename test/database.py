import clickhouse_connect
class ClickhouseDB:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        
        self.client = clickhouse_connect.get_client(host=host, port=port, username=user, password=password, database=database)
        
    def insert_df(self, table_name, df, columns=None):
        if columns is None:
            columns = list(df.columns)


        self.client.insert_df(table=table_name, df=df, column_names=columns)
        return len(df.index)
    
    def drop(self, table_name, condition):
        row = self.client.query(f"""
                DELETE FROM {table_name}
                WHERE {condition}""")
        return row
    