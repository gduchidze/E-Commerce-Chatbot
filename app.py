from flask import Flask, request, jsonify
from chatbot import recommend_products, get_product_details

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if 'details' in user_input.lower():
        product_name = user_input.split('details about ')[-1]
        response = get_product_details(product_name)
    else:
        response = recommend_products(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
if __name__ == '__main__':
    app.run(debug=True, port=5001)

#%%
