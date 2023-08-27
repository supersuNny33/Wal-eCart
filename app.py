from flask import Flask, render_template, request, redirect, jsonify
from pymongo import MongoClient
from bson.objectid import ObjectId
import folium
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

client = MongoClient('mongodb+srv://supersuNny33:zNV2hstjTIIPDAfn@dtg.bsx7n7j.mongodb.net/product_database')
db = client['product_database']
collection = db['products']


shopping_list_collection = db['shopping_list']  # New collection for shopping list
walmart_collection = db['walmart']  # Collection for Walmart locations
barcode_collection = db['barcode']

@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/product_detail')
def product_detail():
    return render_template('barcode.html')

@app.route("/")
def index():
    return render_template("Homepageindex.html")

@app.route('/card')
def cart_index():
    return render_template('cards.html')

@app.route('/list')
def list_index():
    return render_template('shopping_list.html.html')

@app.route('/search', methods=['POST'])
def search():
    keyword = request.form.get('keyword')
    sort_by = request.form.get('sort_by', 'product_name')  # Default sorting by product name
    products = collection.find({'Product_Name': {'$regex': keyword, '$options': 'i'}}).sort(sort_by)
    return render_template('cards.html', products=products, keyword=keyword)

@app.route('/add_to_list', methods=['GET','POST'])
def add_to_list():
    product_id = request.form.get('product_id')
    product = collection.find_one({'_id': ObjectId(product_id)})
    if product:
        shopping_list_collection.insert_one(product)
    return redirect('/shopping_list')
    
@app.route('/barcode', methods=['POST'])
def barcode():
    entered_barcode = request.form['barcode']
    product = barcode_collection.find_one({'barcode': int(entered_barcode)})
    if product:
        return jsonify({'product': render_template('product_detail.html', product=product)})
    else:
        return jsonify({'product': None})

@app.route('/shopping_list')
def shopping_list():
    shopping_list = list(shopping_list_collection.find().sort([('Aisle', 1), ('Shelf', 1)]))
    
    total_price = sum(product['Sale_Price'] for product in shopping_list)
    
    return render_template('shopping_list.html', shopping_list=shopping_list, total_price=total_price)


@app.route('/delete/<string:item_id>')
def delete_item(item_id):
    shopping_list_collection.delete_one({'_id': ObjectId(item_id)})
    return redirect('/shopping_list')

@app.route('/clear_all', methods=['POST'])
def clear_all():
    shopping_list_collection.delete_many({})
    return redirect('/shopping_list')

@app.route('/walmart_map')
def walmart_map_view():
    # Query Walmart locations from the database
    walmart_locations = walmart_collection.find({}, {
        '_id': 0, 'name': 1, 'street_address': 1, 'city': 1, 'state': 1, 'zip_code': 1,
        'phone_number': 1, 'Open/Closed': 1, 'Pharmacy Open': 1,
        'Online Grocery Pickup Service Offered': 1, 'Online Grocery Pickup Status': 1,
        'Grocery Delivery Service Offered': 1, 'Grocery Delivery Service Status': 1,
        'latitude': 1, 'longitude': 1  # Add latitude and longitude fields to the projection
    })
    
    # Create a Folium map centered on a specific location (e.g., city center)
    map = folium.Map(location=[35.463405, -97.622266], zoom_start=12)  # Oklahoma City
    
    # Add markers for Walmart locations
    
    for walmart in walmart_locations:
        popup_html = f"<div class='custom-popup'><b>{walmart['name']}</b><br>"
        popup_html += f"Address: {walmart['street_address']}, {walmart['city']}, {walmart['state']} {walmart['zip_code']}<br>"
        popup_html += f"Phone: {walmart['phone_number']}<br>"
        popup_html += f"Open/Closed: {walmart['Open/Closed']}<br>"
        popup_html += f"Pharmacy Open: {walmart['Pharmacy Open']}<br>"
        popup_html += f"Online Grocery Pickup: {walmart['Online Grocery Pickup Service Offered']} ({walmart['Online Grocery Pickup Status']})<br>"
        popup_html += f"Grocery Delivery: {walmart['Grocery Delivery Service Offered']} ({walmart['Grocery Delivery Service Status']})</div>"
        
        # Convert latitude and longitude strings to floats
        latitude = float(walmart['latitude'])
        longitude = float(walmart['longitude'])
        
        folium.Marker(
            location=[latitude, longitude],
            popup=folium.Popup(popup_html, max_width=800),  # Adjust max_width as needed
            icon=folium.Icon(icon="info-sign")
        ).add_to(map)
    
    # Convert the map to HTML and pass it to the template
    map_html = map._repr_html_()
    
    # Return the template with the map
    return render_template('stores.html', map_html=map_html)

user_item_matrix = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
])

# Calculate item-item similarity using cosine similarity
item_sim_matrix = cosine_similarity(user_item_matrix.T)

product_id_to_index = {}
for index, product in enumerate(collection.find()):
    product_id_to_index[str(product['_id'])] = index

@app.route('/recommendations')
def recommendations():
    # Get user's shopping list (you can replace this with actual user data)
    user_shopping_list = shopping_list_collection.find()

    # Get indices of items in the shopping list using the mapping
    shopping_list_indices = [product_id_to_index[str(item['_id'])] for item in user_shopping_list]

    # Calculate recommendations based on item-item similarity
    recommendations = np.zeros(user_item_matrix.shape[1])  # Initialize recommendation scores

    for item_index in shopping_list_indices:
        if item_index >= user_item_matrix.shape[0]:
            continue  # Skip if item_index is out of bounds
        # Calculate item-item similarity with other items in the user_item_matrix
        item_similarity = cosine_similarity([user_item_matrix[item_index]], user_item_matrix)
        
        recommendations += item_similarity[0]  # Add similarity scores to recommendations

    # Sort and get top N recommended items
    num_recommendations = 5
    recommended_indices = np.argsort(recommendations)[::-1][:num_recommendations]

    # Retrieve actual product information based on indices
    recommended_products = []
    for item_index in recommended_indices:
        # Retrieve the product using the index-to-product_id mapping
        product_id = next((key for key, value in product_id_to_index.items() if value == item_index), None)
        if product_id:
            recommended_product = collection.find_one({'_id': ObjectId(product_id)})
            if recommended_product:
                recommended_products.append(recommended_product)

    return render_template('recommendations.html', recommended_products=recommended_products)


# Load your trained model
df = pd.read_csv('new_dummy_customer_data.csv')

# Data Preprocessing
df_encoded = pd.get_dummies(df, columns=['Gender', 'SentimentOfReviews', 'ParticipatedInCampaign', 'ResponseToEmail'], drop_first=True)

# Define relevant features
relevant_features = ['Age', 'TotalSpending', 'RecencyOfPurchase', 'NumItemsPerOrder', 'AvgProductPrice',
                     'TimeOnWebsite', 'NumAbandonedCarts', 'NumCustomerServiceCalls', 'AvgRating',
                     'Gender_Male', 'SentimentOfReviews_Neutral', 'ParticipatedInCampaign_Yes']

X = df_encoded[relevant_features]
y = df_encoded['Churned']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    churn_result = None

    if request.method == 'POST':
        new_customer = {
            'Age': int(request.form['Age']),
            'TotalSpending': float(request.form['TotalSpending']),
            'RecencyOfPurchase': int(request.form['RecencyOfPurchase']),
            'NumItemsPerOrder': int(request.form['NumItemsPerOrder']),
            'AvgProductPrice': float(request.form['AvgProductPrice']),
            'TimeOnWebsite': float(request.form['TimeOnWebsite']),
            'NumAbandonedCarts': int(request.form['NumAbandonedCarts']),
            'NumCustomerServiceCalls': int(request.form['NumCustomerServiceCalls']),
            'AvgRating': float(request.form['AvgRating']),
            'Gender': request.form['Gender'],
            'SentimentOfReviews': request.form['SentimentOfReviews'],
            'ParticipatedInCampaign': request.form['ParticipatedInCampaign']
        }

        new_customer_data = pd.DataFrame(new_customer, index=[0])
        new_customer_encoded = pd.get_dummies(new_customer_data, columns=['Gender', 'SentimentOfReviews', 'ParticipatedInCampaign'], drop_first=True)
        new_customer_data_final = new_customer_encoded.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(new_customer_data_final)
        churn_probability = model.predict_proba(new_customer_data_final)[:, 1]

        if prediction == 1:
            churn_result = f"The new customer is likely to churn with a churn probability of {churn_probability[0]:.2f}"
        else:
            churn_result = f"The new customer is likely to stay with a churn probability of {churn_probability[0]:.2f}"

    return render_template('churn.html', churn_result=churn_result)


if __name__ == '__main__':
    app.run(debug=True)
