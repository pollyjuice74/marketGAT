class Trader():
    def __init__(self, account):
        self.account = account if account not None else Account()
