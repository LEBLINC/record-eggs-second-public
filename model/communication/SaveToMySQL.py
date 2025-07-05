import pymysql

class SaveToMySQL:
    def __init__(self, config):
        self.host = 'localhost'
        self.port = 3306
        self.user = 'root'  # 如有需要请修改
        self.password = '123456'  # 如有需要请修改
        self.db = 'wenshi_eggs_record'
        self.table = config.get('table', 'duckdata1')  # 优先用配置传入的表名
        self.conn = pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            db=self.db,
            charset='utf8mb4'
        )
        self.cursor = self.conn.cursor()

    def save(self, send_data):
        # 解析二维码内容
        qr_code = str(send_data['cage_id'])
        if '/' in qr_code:
            cage_str, cx_wb_str = qr_code.split('/', 1)
            try:
                cage = int(cage_str)
            except:
                cage = 0
            try:
                cx_wb = int(cx_wb_str)
            except:
                cx_wb = 0
        else:
            cage = 0
            cx_wb = 0

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
        self.cursor.execute(sql, values)
        self.conn.commit() 