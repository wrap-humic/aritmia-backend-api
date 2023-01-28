from . import db
from collections import OrderedDict


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255))
    password = db.Column(db.Text)
    nama_lengkap = db.Column(db.String(255))

    def to_dict(self):
        return OrderedDict([
            ('id', self.id),
            ('email', self.email),
            ('nama_lengkap', self.nama_lengkap)
        ])
