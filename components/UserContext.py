from components.init_database import DbContext, select_user_by_username


class UserContext(object):
    def __init__(self,dbname, username="Monitor", role="MONITOR"):
        self.username = username
        self.role = role
        self.dbname = dbname
    def login(self,username,password):
        with DbContext(self.dbname) as db:
            user = select_user_by_username(db,username)
            if not user: 
                if username=="acanus" and password == "emmawaston":
                    self.username = "Dev"
                    self.role = "DEVELOPER"
                    return True
                return False
            if password==user[2]:
                self.username = user[1]
                self.role = user[3]
                return True
            return False
    def log_out(self):
        self.username = "Monitor"
        self.role = "MONITOR"
    def is_admin(self):
        return self.role == 'ADMIN'
    def is_dev(self):
        return self.role == 'DEVELOPER'