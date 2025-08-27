import duckdb

def get_duck():
    # in-memory DB is perfect for CSV uploads
    return duckdb.connect(database=":memory:", read_only=False)
