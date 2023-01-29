from . import db
from collections import OrderedDict
from sqlalchemy.types import Enum


class Record(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    id_pasien = db.Column(db.Integer)
    path = db.Column(db.Text)
    tanggal_record = db.Column(db.Date)

    def to_dict(self):
        return OrderedDict([
            ('id', self.id),
            ('id_pasien', self.id_pasien),
            ('path', self.path),
            ('tanggal_record', self.tanggal_record),
        ])
