from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/", methods=['GET'])
def two_input():
    return render_template('two_input.html')


@app.route("/two_result", methods=['POST'])
def two_result():
    value1 = request.form.get('value1')
    value2 = request.form.get('value2')
    sum = int(value1) + int(value2)

    return render_template('two_result.html', value1=value1, value2=value2, sum=sum)


if __name__ == "__main__":
    app.run()