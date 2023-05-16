import cx_Oracle

# データベースへの接続
def connect(user, password, dsn):
    """
    DB接続を確立する

    :param user: ユーザ名
    :param password: パスワード
    :param dsn: 接続文字列
    :return: connectionオブジェクト
    """
    conn = cx_Oracle.connect(user, password, dsn)
    return conn

# データベースからの切断
def disconnect(conn):
    """
    DB接続を解除する

    :param conn: connectionオブジェクト
    """
    conn.close()

# SQLファイルからSQL文を取得する
def get_sql_from_file(file_path, sql_id):
    """
    SQLファイルからSQL文を取得する

    :param file_path: SQLファイルのパス
    :param sql_id: SQL文のID
    :return: SQL文
    """
    with open(file_path) as f:
        sql_dict = {}
        for line in f:
            if line.startswith("--"):
                sql_id = line.strip("--").strip()
            else:
                sql_dict[sql_id] = sql_dict.get(sql_id, "") + line
    return sql_dict.get(sql_id)

# SELECT文を実行する
def select(conn, file_path, sql_id, params=None):
    """
    SELECT文を実行する

    :param conn: connectionオブジェクト
    :param file_path: SQLファイルのパス
    :param sql_id: SQL文のID
    :param params: SQL文のパラメータ
    :return: 結果のリスト
    """
    cursor = conn.cursor()
    sql = get_sql_from_file(file_path, sql_id)
    if params:
        cursor.execute(sql, params)
    else:
        cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    return result

# INSERT文を実行する
def insert(conn, table_name, model):
    """
    INSERT文を実行する

    :param conn: connectionオブジェクト
    :param table_name: テーブル名
    :param model: モデル
    """
    cursor = conn.cursor()
    columns = list(model.keys())
    values = list(model.values())
    placeholders = ",".join(["%s"] * len(columns))
    sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"
    cursor.execute(sql, values)
    conn.commit()
    cursor.close()

# UPDATE文を実行する
def update(conn, table_name, model, condition):
    """
    UPDATE文を実行する

    :param conn: connectionオブジェクト
    :param table_name: テーブル名
    :param model: モデル
    :param condition: 更新条件
    """
    cursor = conn.cursor()
    set_clause = ",".join([f"{column} = %s" for column in model.keys()])
    sql = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"
    cursor.execute(sql, list(model.values()))
    conn.commit()
    cursor.close()

# DELETE文を実行する
def delete(conn, table_name, condition):
    """
    DELETE文を実行する

    :param conn: connectionオブジェクト
    :param table_name: テーブル名
    :param condition: 削除条件
    """
    cursor = conn.cursor()
    sql = f"DELETE FROM {table_name} WHERE {condition}"
    cursor.execute(sql)
    conn.commit()
    cursor.close()
