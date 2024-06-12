import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv('sample_data_10k.csv')

# Combine relevant columns for product description
data['description'] = data['About Product'].fillna('') + ' ' + \
                      data['Product Specification'].fillna('') + ' ' + \
                      data['Product Details'].fillna('')

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(data['description'])

def recommend_products(user_request):
    user_tfidf = vectorizer.transform([user_request])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[-5:][::-1]
    recommended_products = data.iloc[top_indices]

    response = "Here are some products that match your request:\n"
    for idx, product in recommended_products.iterrows():
        response += f"{product['Product Name']} - {product['About Product']}\n"
        response += f"Price: {product['Selling Price']}\n"
        response += "------------\n"

    return response

def get_product_details(product_name):
    product = data[data['Product Name'].str.contains(product_name, case=False)].iloc[0]
    details = f"Product: {product['Product Name']}\n"
    details += f"Description: {product['About Product']}\n"
    details += f"Price: {product['Selling Price']}\n"
    details += f"Category: {product['Category']}\n"
    return details

def main():
    print("Welcome to the eCommerce Chatbot!")
    print("Type 'exit' to end the conversation.")

    while True:
        user_input = input("\nHow can I help you today? ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        if 'details about' in user_input.lower():
            product_name = user_input.split('details about ')[-1]
            response = get_product_details(product_name)
        else:
            response = recommend_products(user_input)

        print(response)

if __name__ == "__main__":
    main()

#%%

#%%

#%%
