import json

from flask import Flask, render_template, request, jsonify

from demo.Demo import Demo
from demo.demo_cli import config
from squad.demo_prepro import prepro

########################################################################
### These are only used for displaying sample paragraphs and questions
########################################################################

shared = json.load(open("data/squad_demo/shared_test.json", "r"))
contextss = [""]
context_questions = [""]
for i in range(len(shared['contextss'])):
    j = 1 if i == 0 else 0
    contextss.append(shared["contextss"][i][j])
    context_questions.append(shared['context_questions'][i][j])
titles = ["Write own paragraph"] + shared["titles"]

########################################################################
### Demo loads the pretrained weight
########################################################################

# construct the flask app
app = Flask(__name__, static_folder="static", template_folder="templates")

# get config and construct the Demo
demo = Demo(config)


def getTitle(ai):
    return titles[ai]


def getPara(rxi):
    return contextss[rxi[0]][rxi[1]]


def getAnswer(paragraph, question):
    pq_prepro = prepro(paragraph, question)
    if len(pq_prepro['x']) > 1000:
        return "[Error] Sorry, the number of words in paragraph cannot be more than 1000."
    if len(pq_prepro['q']) > 100:
        return "[Error] Sorry, the number of words in question cannot be more than 100."
    return demo.run(pq_prepro)


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/select', methods=['GET', 'POST'])
def select():
    # paragraph_id = request.args.get('paragraph_id', type=int)
    # rxi = [paragraph_id, 0]
    # paragraph = getPara(rxi)
    # return jsonify(result=paragraph)
    return jsonify(result={"titles": titles, "contextss": contextss, "context_questions": context_questions})


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    paragraph = request.args.get('paragraph')
    question = request.args.get('question')
    answer = getAnswer(paragraph, question)
    return jsonify(result=answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="1995", threaded=True)
