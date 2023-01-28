from flask import jsonify
from collections import OrderedDict


def response_helper(status_code, message, data):
    response = jsonify(OrderedDict(
        status_code=status_code, message=message, data=data))
    response.status_code = status_code
    return response
