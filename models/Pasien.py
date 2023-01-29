from . import db
from collections import OrderedDict
from sqlalchemy.types import Enum


class Pasien(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama = db.Column(db.String(255))
    jenis_kelamin = db.Column(
        Enum('LAKI-LAKI', 'PEREMPUAN', name='jenis_kelamin'))
    umur = db.Column(db.Integer)
    tanggal_lahir = db.Column(db.Date)
    kondisi_kesehatan = db.Column(db.String(255))

    def to_dict(self):
        return OrderedDict([
            ('id', self.id),
            ('nama', self.nama),
            ('jenis_kelamin', self.jenis_kelamin),
            ('umur', self.umur),
            ('tanggal_lahir', self.tanggal_lahir),
            ('kondisi_kesehatan', self.kondisi_kesehatan),
        ])
