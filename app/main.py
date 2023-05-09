from flask import Flask

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return 'Server is up and running!', 200

@app.route('/GetForecast', methods=['GET'])
def get_forecast():
    # your forecast code here
    return 'Here is your forecast!', 200

if __name__ == '__main__':
    app.run(debug=True)
