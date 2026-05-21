import pymysql

class SaveToMySQL:
    def __init__(self, config):
        self.host = config.get('host', 'localhost')
        self.port = int(config.get('port', 3306))
        self.user = config.get('user', 'root')
        self.password = config.get('password', '123456')
        self.db = config.get('db', 'wenshi_eggs_record')
        self.table = config.get('table', 'duckdata1')
        self.include_img = config.get('include_img', True)

        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                charset='utf8mb4'
            )
            self.cursor = self.conn.cursor()
            print(f"数据库连接成功: {self.host}:{self.port} - {self.db}")
        except Exception as e:
            print(f"数据库连接失败 ({self.host}): {e}")
            self.conn = None
            self.cursor = None

    def save(self, send_data):
        if not self.conn or not self.cursor:
            # 尝试重连
            try:
                self.conn = pymysql.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db=self.db,
                    charset='utf8mb4'
                )
                self.cursor = self.conn.cursor()
            except Exception:
                return

        # 解析二维码内容
        qr_code = str(send_data.get('cage_id', ''))
        cage = 0
        cx_wb = 0
        
        if '/' in qr_code:
            cage_str, cx_wb_str = qr_code.split('/', 1)
            try:
                cage = int(cage_str)
            except:
                pass
            try:
                cx_wb = int(cx_wb_str)
            except:
                pass

        try:
            if self.include_img:
                sql = f"""
                INSERT INTO {self.table} 
                (id_code, cx_wb, cage, centrydate, ge, je, se, be, de, note, img)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    qr_code,      # id_code
                    cx_wb,        # cx_wb
                    cage,         # cage
                    send_data['record_time'],  # centrydate
                    send_data['egg_num'],      # ge
                    0, 0, 0, 0, '',            # je, se, be, de, note
                    send_data['frame_path']    # img
                )
            else:
                # 不包含img列
                sql = f"""
                INSERT INTO {self.table} 
                (id_code, cx_wb, cage, centrydate, ge, je, se, be, de, note)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    qr_code,      # id_code
                    cx_wb,        # cx_wb
                    cage,         # cage
                    send_data['record_time'],  # centrydate
                    send_data['egg_num'],      # ge
                    0, 0, 0, 0, ''             # je, se, be, de, note
                )
            
            self.cursor.execute(sql, values)
            self.conn.commit()
            print(f"数据保存成功 ({self.host}): {qr_code}")
        except Exception as e:
            print(f"数据保存异常 ({self.host}): {e}")
            # 可能是连接断开，尝试重连标记?
            try:
                self.conn.ping(reconnect=True)
            except:
                pass 


class SaveQrToMySQL:
    """保存二维码抓拍图片（本地库）。"""
    def __init__(self, config):
        self.host = config.get('host', 'localhost')
        self.port = int(config.get('port', 3306))
        self.user = config.get('user', 'root')
        self.password = config.get('password', '123456')
        self.db = config.get('db', 'wenshi_eggs_record')
        self.table = config.get('table', 'qr_code_images')
        self.ensure_table = bool(config.get('ensure_table', True))

        self.conn = None
        self.cursor = None
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                charset='utf8mb4'
            )
            self.cursor = self.conn.cursor()
            if self.ensure_table:
                self._ensure_table()
            print(f"二维码库连接成功: {self.host}:{self.port} - {self.db}")
        except Exception as e:
            print(f"二维码库连接失败 ({self.host}): {e}")

    def _ensure_table(self):
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            id_code VARCHAR(64) NOT NULL,
            img_path VARCHAR(512),
            scan_time DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            self.cursor.execute(sql)
            try:
                self.cursor.execute(f"CREATE INDEX idx_{self.table}_id_code ON {self.table}(id_code)")
            except Exception:
                pass
            self.conn.commit()
        except Exception:
            pass

    def _reconnect(self):
        if self.conn and self.cursor:
            return True
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db=self.db,
                charset='utf8mb4'
            )
            self.cursor = self.conn.cursor()
            if self.ensure_table:
                self._ensure_table()
            return True
        except Exception:
            return False

    def get_last_scan_time(self, id_code: str):
        if not id_code:
            return None
        if not self._reconnect():
            return None
        try:
            sql = f"SELECT scan_time FROM {self.table} WHERE id_code=%s ORDER BY scan_time DESC LIMIT 1"
            self.cursor.execute(sql, (id_code,))
            row = self.cursor.fetchone()
            if row and row[0]:
                if hasattr(row[0], 'timestamp'):
                    return row[0].timestamp()
                if isinstance(row[0], str):
                    try:
                        import datetime
                        dt = datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                        return dt.timestamp()
                    except Exception:
                        return None
                return row[0]
        except Exception:
            return None
        return None

    def save_qr_image(self, id_code: str, img_path: str, record_time: str):
        if not id_code:
            return
        if not self._reconnect():
            return
        try:
            # 保留历史记录：每次插入一条
            sql = f"INSERT INTO {self.table} (id_code, img_path, scan_time) VALUES (%s, %s, %s)"
            self.cursor.execute(sql, (id_code, img_path, record_time))
            self.conn.commit()
            print(f"二维码抓拍保存成功: {id_code}")
        except Exception as e:
            print(f"二维码抓拍保存异常: {e}")