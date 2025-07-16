import enum
import logging
import pickle
import pandas as pd

from .cache import PersistentKeyValueCache, DelayedUpdateHook

log = logging.getLogger(__name__)


class MySQLPersistentKeyValueCache(PersistentKeyValueCache):
    """
    Can cache arbitrary values in a MySQL database.
    The keys are always strings at the database level, i.e. if a key is not a string, it is converted to a string using str().
    """

    class ValueType(enum.Enum):
        """
        The value type to use within the MySQL database.
        Note that the binary BLOB types can be used for all Python types that can be pickled, so the lack
        of specific types (e.g. for strings) is not a problem.
        """
        # enum values are (SQL data type, isCachedValuePickled)
        DOUBLE = ("DOUBLE", False)
        BLOB = ("BLOB", True)
        """
        for Python data types whose pickled representation is up to 64 KB
        """
        MEDIUMBLOB = ("MEDIUMBLOB", True)
        """
        for Python data types whose pickled representation is up to 16 MB
        """

    def __init__(self, host: str, db: str, user: str, pw: str, value_type: ValueType, table_name="cache",
        connect_params: dict | None = None, in_memory=False, max_key_length: int = 255, port=3306):
        """
        :param host:
        :param db:
        :param user:
        :param pw:
        :param value_type: the type of value to store in the cache
        :param table_name:
        :param connect_params: additional parameters to pass to the pymysql.connect() function (e.g. ssl, etc.)
        :param in_memory:
        :param max_key_length: maximal length of the cache key string (keys are always strings) stored in the DB
            (i.e. the MySQL type is VARCHAR[max_key_length])
        :param port: the MySQL server port to connect to
        """
        import pymysql
        if connect_params is None:
            connect_params = {}
        self._connect = lambda: pymysql.connect(host=host, database=db, user=user, password=pw, port=port, autocommit=True,
            **connect_params)
        self._conn = self._connect()
        self.table_name = table_name
        self.max_key_length = max_key_length

        cache_value_sql_type, self.is_cache_value_pickled = value_type.value

        cursor = self._conn.cursor()
        cursor.execute(f"SHOW TABLES;")
        if table_name not in [r[0] for r in cursor.fetchall()]:
            log.debug(f"Creating table {table_name}")
            cursor.execute(f"CREATE TABLE {table_name} (cache_key VARCHAR({self.max_key_length}) PRIMARY KEY, "
                           f"cache_value {cache_value_sql_type});")
        cursor.close()

        self._in_memory_df = None if not in_memory else self._load_table_to_data_frame()

    def _cursor(self):
        try:
            self._conn.ping(reconnect=True)
        except Exception as e:
            log.error(f"Error while pinging MySQL server: {e}; Reconnecting ...")
            self._conn = self._connect()
        return self._conn.cursor()

    def _load_table_to_data_frame(self):
        df = pd.read_sql(f"SELECT * FROM {self.table_name};", con=self._conn, index_col="cache_key")
        if self.is_cache_value_pickled:
            df["cache_value"] = df["cache_value"].apply(pickle.loads)
        return df

    def set(self, key, value):
        key = str(key)
        if len(key) > self.max_key_length:
            raise ValueError(f"Key too long, maximal key length is {self.max_key_length}")
        cursor = self._cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE cache_key=%s", (key,))
        stored_value = pickle.dumps(value) if self.is_cache_value_pickled else value
        if cursor.fetchone()[0] == 0:
            from pymysql.err import IntegrityError
            try:
                cursor.execute(f"INSERT INTO {self.table_name} (cache_key, cache_value) VALUES (%s, %s)",
                    (key, stored_value))
            except IntegrityError as e:
                if e.args[0] == 1062:  # Duplicate entry
                    # This can only happen when the user is inserting the same value almost simultaneously (race condition)
                    args = list(e.args)
                    args[1] = f"{args[1]}; The duplicate entry is due to quasi-simultaneous insertions for the same key; " \
                              "Check your application logic!"
                    raise IntegrityError(*args)
                else:
                    raise
        else:
            cursor.execute(f"UPDATE {self.table_name} SET cache_value=%s WHERE cache_key=%s", (stored_value, key))
        cursor.close()
        if self._in_memory_df is not None:
            self._in_memory_df["cache_value"][str(key)] = value

    def get(self, key):
        value = self._get_from_in_memory_df(key)
        if value is None:
            value = self._get_from_table(key)
        return value

    def _get_from_table(self, key):
        cursor = self._cursor()
        cursor.execute(f"SELECT cache_value FROM {self.table_name} WHERE cache_key=%s", (str(key),))
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return None
        stored_value = row[0]
        value = pickle.loads(stored_value) if self.is_cache_value_pickled else stored_value
        return value

    def _get_from_in_memory_df(self, key):
        if self._in_memory_df is None:
            return None
        try:
            return self._in_memory_df["cache_value"][str(key)]
        except Exception as e:
            log.debug(f"Unable to load value for key {str(key)} from in-memory dataframe: {e}")
            return None
