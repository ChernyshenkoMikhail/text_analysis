from flask import Flask, request, send_file, render_template
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import os
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file and file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
        word_counts = analyze_text(text)
        lda_results = perform_lda(text)
        sentiment_score = analyze_sentiment(text)
        lexical_diversity, avg_sentence_length, total_sentences = analyze_text_statistics(text)
        
        # Сохранение результатов в один Excel-файл
        results_file = 'analysis_results.xlsx'
        save_results_to_excel(word_counts, lda_results, sentiment_score, 
                            lexical_diversity, avg_sentence_length, total_sentences, 
                            results_file)
        
        return send_file(results_file, as_attachment=True)
    else:
        return "Only .txt files are allowed"

def analyze_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords.words('russian')]
    word_counts = Counter(words)
    return word_counts

def perform_lda(text):
    # Преобразование текста в формат для LDA
    vectorizer = CountVectorizer(stop_words=russian_stopwords)
    doc_term_matrix = vectorizer.fit_transform([text])

    # Модель LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(doc_term_matrix)
    
    generate_wordcloud(text)
    
    # Получение тем
    words = vectorizer.get_feature_names_out()
    topics = {}
    for idx, topic in enumerate(lda.components_):
        topics[f'Topic {idx+1}'] = [words[i] for i in topic.argsort()[-5:]]
    return topics

def analyze_sentiment(text):
    # Проведение сентимент-анализа с помощью TextBlob
    blob = TextBlob(text)
    return blob.sentiment.polarity

def analyze_text_statistics(text):
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    
    if total_sentences > 0:
        avg_sentence_length = sum(len(word_tokenize(sentence)) for sentence in sentences) / total_sentences
    else:
        avg_sentence_length = 0
    
    words = word_tokenize(text)
    lexical_diversity = len(set(words)) / len(words) if len(words) > 0 else 0
    
    return lexical_diversity, avg_sentence_length, total_sentences

russian_stopwords = [
    "и", "в", "во", "не", "нет", "на", "то", "что", "как", "это",
    "за", "с", "он", "она", "о", "из", "к", "по", "для", "это"]

def generate_wordcloud(text):
    # Создание облака слов
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=russian_stopwords).generate(text)
    
    # Сохранение облака слов в файл PNG
    wordcloud_path = "wordcloud.png"
    wordcloud.to_file(wordcloud_path)

    # Отображение облака слов
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # Возвращаем путь к файлу, чтобы пользователь мог скачать его
    return wordcloud_path

def save_results_to_excel(word_counts, lda_results, sentiment_score, lexical_diversity, avg_sentence_length, total_sentences, filename):
    
    # Подготовка данных для Excel
    word_counts_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
    lda_results_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lda_results.items()]))
    
    # Создание DataFrame для сентимент-анализа
    sentiment_df = pd.DataFrame({'Sentiment Score': [sentiment_score]})
    
    # Статистические показатели
    stats_df = pd.DataFrame({
        'Lexical Diversity': [lexical_diversity],
        'Average Sentence Length': [avg_sentence_length],
        'Total Sentences': [total_sentences]
    })

    with pd.ExcelWriter(filename) as writer:
        word_counts_df.to_excel(writer, sheet_name='Word Counts', index=False)
        lda_results_df.to_excel(writer, sheet_name='LDA Results', index=False)
        sentiment_df.to_excel(writer, sheet_name='Sentiment Analysis', index=False)
        stats_df.to_excel(writer, sheet_name='Text Statistics', index=False)

if __name__ == '__main__':
    app.run(debug=True)