from flask import Flask, request, jsonify
import sys
sys.path.append('/Users/illiafransua/Documents/Projects/SpamGuardService/spam_guard/lib')

from lib.censure import Censor
from model import SpamDetectionModel


app = Flask(__name__)

model = SpamDetectionModel()
censor_ru = Censor.get(lang='ru')
censor_en = Censor.get(lang='en')

@app.route('/is_spam', methods=['POST'])
def is_spam():
    data = request.get_json()
    text = data.get('text', '').lower()

    if not text:
        return jsonify({'result': False})

    text = ' '.join(text.split())

    is_bad = (
        model.predict(text) or
        not censor_ru.check_line(text).get('is_good', True) or
        not censor_en.check_line(text).get('is_good', True)
    )

    return jsonify({'result': bool(is_bad)})

@app.route('/save_as_spam', methods=['POST'])
def save_as_spam():
    data = request.get_json()
    text = data.get('text', '').lower()

    if not text:
        return jsonify({'result': False})

    text = ' '.join(text.split())

    with open('spam_list.txt', 'a') as file:
        file.write(text + '\n')

    return jsonify({'result': True})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8000)
