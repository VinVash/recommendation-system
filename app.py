from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the model at the start
model_data = joblib.load('svd_model_new.pkl')
u = model_data['U']
sigma = model_data['sigma']
vt = model_data['Vt']
products = model_data['products']
users = model_data['users']
mean_user_rating = model_data['mean_user_rating'].reindex(users).fillna(0)  # Fill NA for any user without ratings
# sigma = np.diag(s)


@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id')
    # user_id is of type str. Convert it to int
    user_id = int(user_id)
    if user_id not in users:
        return jsonify({'error': 'User not found'}), 404
    
    user_idx = list(users).index(user_id)
    predicted_ratings = np.dot(np.dot(u[user_idx, :], sigma), vt) + mean_user_rating.loc[user_id]
    top_products = np.argsort(predicted_ratings)[::-1][:10]  # get top 10 products
    recommended_product_ids = products[top_products]

    return jsonify({'user_id': user_id, 'recommended_products': recommended_product_ids.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
