import json

class PreprocessJSON(object):

    def openJSON(self, filename):
        """open JSON file to object"""
        with open(filename, 'r') as f:
            json_obj = json.load(f)

        return json_obj

    def encodeJSON(self, json_obj):
        """encode JSON file to bytes --> return ourbyte"""
        ourbyte=b''
        ourbyte = json.dumps(json_obj).encode("utf-8")

        return ourbyte

    def writeJSON(self, filename, json_obj):
        """Linear Actuator IP, PORT, location dict, move_z"""
        with open(filename, "w") as json_file:
            json.dumps(json_obj, json_file)


