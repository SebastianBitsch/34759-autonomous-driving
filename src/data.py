


class KittyData:
    def __init__(self, data):
        self.data = data

    def get(self):
        return self.data

    def get_kitty(self, kitty_id):
        return self.data[kitty_id]