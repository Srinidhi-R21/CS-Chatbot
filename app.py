from flask import Flask, render_template, request, jsonify, redirect, url_for
from enhanced_chatbot import CustomerSupportBot

app = Flask(__name__)
chatbot = CustomerSupportBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        question = request.form.get('question')
        answer = request.form.get('answer')
        category = request.form.get('category')
        
        if question and answer:
            success = chatbot.kb.add_entry(question, answer, category)
            if success:
                return redirect(url_for('admin'))
    
    return render_template('admin.html', categories=chatbot.kb.categories)

if __name__ == '__main__':
    app.run(debug=True) 