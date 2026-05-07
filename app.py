from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

documents = []
file_names = []

folder_path = 'documents'

for file in os.listdir(folder_path):

    with open(
        os.path.join(folder_path, file),
        'r',
        encoding='utf-8'
    ) as f:

        documents.append(f.read())

        file_names.append(file)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)


@app.route('/', methods=['GET', 'POST'])
def home():

    results = []

    query = ''

    if request.method == 'POST':

        query = request.form['query']

        query_vector = vectorizer.transform([query])

        similarity = cosine_similarity(
            query_vector,
            tfidf_matrix
        )

        scores = similarity.flatten()

        ranked_results = scores.argsort()[::-1]

        for i in ranked_results:

            if scores[i] > 0:

                results.append({
                    'file': file_names[i],
                    'content': documents[i],
                    'score': round(scores[i], 2)
                })

    return render_template(
        'index.html',
        results=results,
        query=query
    )


if __name__ == '__main__':
    app.run(debug=True)