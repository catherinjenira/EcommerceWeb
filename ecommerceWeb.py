from flask import Flask, render_template_string, request, jsonify
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from math import sqrt

app = Flask(__name__)

# Sample product database
products = [
    {
        "id": 1,
        "name": "Wireless Bluetooth Headphones",
        "category": "Electronics",
        "price": 79.99,
        "rating": 4.5,
        "description": "High-quality wireless headphones with noise cancellation",
        "tags": ["wireless", "bluetooth", "headphones", "audio", "music"],
        "image": "https://www.bhphotovideo.com/images/images2000x2000/sony_wh1000xm2_b_1000x_wireless_noise_canceling_headphones_1361028.jpg"
    },
    {
        "id": 2,
        "name": "Smartphone XYZ",
        "category": "Electronics",
        "price": 699.99,
        "rating": 4.3,
        "description": "Latest smartphone with advanced camera and fast processor",
        "tags": ["smartphone", "mobile", "android", "camera", "tech"],
        "image": "https://tse3.mm.bing.net/th/id/OIP.3F1uHClGROs9kuyaN3HnmQHaLl?cb=ucfimg2ucfimg=1&rs=1&pid=ImgDetMain&o=7&rm=3"
    },
    
       
    {
        "id": 4,
        "name": "Laptop Pro",
        "category": "Electronics",
        "price": 1299.99,
        "rating": 4.6,
        "description": "Powerful laptop for work and gaming with high-performance specs",
        "tags": ["laptop", "computer", "gaming", "work", "tech"],
        "image": "https://th.bing.com/th/id/OIP.dgsHEv95AMaaNNVM_TTw1QHaEj?w=297&h=183&c=7&r=0&o=7&cb=ucfimg2&dpr=1.3&pid=1.7&rm=3&ucfimg=1"
    },
    {
        "id": 5,
        "name": "Coffee Maker",
        "category": "Home",
        "price": 89.99,
        "rating": 4.2,
        "description": "Automatic coffee maker for brewing perfect coffee every morning",
        "tags": ["coffee", "kitchen", "home", "appliance", "brew"],
        "image": "https://tse4.mm.bing.net/th/id/OIP.A8vhT8zueZXfe1BLkhwn3wHaLC?cb=ucfimg2ucfimg=1&rs=1&pid=ImgDetMain&o=7&rm=3"
    },
    {
        "id": 6,
        "name": "Fitness Tracker",
        "category": "Electronics",
        "price": 49.99,
        "rating": 4.0,
        "description": "Track your steps, heart rate, and sleep patterns",
        "tags": ["fitness", "tracker", "health", "wearable", "sports"],
        "image": "https://th.bing.com/th/id/OIP.qU0Y5Q3Tms8CoFtSIcUdsgHaK3?w=131&h=192&c=7&r=0&o=7&cb=ucfimg2&dpr=1.3&pid=1.7&rm=3&ucfimg=1"
    },
    {
        "id": 7,
        "name": "Backpack",
        "category": "Fashion",
        "price": 39.99,
        "rating": 4.4,
        "description": "Durable backpack for school, work, and travel",
        "tags": ["backpack", "bag", "travel", "school", "fashion"],
        "image": ""
    },
    {
        "id": 8,
        "name": "Desk Lamp",
        "category": "Home",
        "price": 29.99,
        "rating": 4.1,
        "description": "LED desk lamp with adjustable brightness for reading and work",
        "tags": ["lamp", "lighting", "home", "office", "desk"],
        "image": "https://th.bing.com/th/id/OIP.h3Robvsuu4wrfg0DZRgzIAHaHa?w=183&h=183&c=7&r=0&o=7&cb=ucfimg2&dpr=1.3&pid=1.7&rm=3&ucfimg=1"
    }
]

class ProductSearch:
    def __init__(self, products):
        self.products = products
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._build_search_index()
    
    def _build_search_index(self):
        """Build TF-IDF vectors for product search"""
        product_texts = []
        for product in self.products:
            text = f"{product['name']} {product['description']} {product['category']} {' '.join(product['tags'])}"
            product_texts.append(text)
        
        if product_texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(product_texts)
        else:
            self.tfidf_matrix = None
    
    def search_products(self, query, category=None, max_price=None, min_rating=None):
        """Search products based on query and filters"""
        results = []
        
        # Text-based search using TF-IDF
        if query and self.tfidf_matrix is not None:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            for i, similarity in enumerate(similarities):
                product = self.products[i].copy()
                product['similarity_score'] = similarity
                results.append(product)
            
            # Sort by similarity score
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
        else:
            results = [product.copy() for product in self.products]
        
        # Apply filters
        filtered_results = []
        for product in results:
            if category and product['category'].lower() != category.lower():
                continue
            if max_price and product['price'] > max_price:
                continue
            if min_rating and product['rating'] < min_rating:
                continue
            filtered_results.append(product)
        
        return filtered_results
    
    def get_recommendations(self, product_id, num_recommendations=4):
        """Get product recommendations based on similarity"""
        if self.tfidf_matrix is None:
            return []
        
        # Find the target product
        target_idx = None
        for i, product in enumerate(self.products):
            if product['id'] == product_id:
                target_idx = i
                break
        
        if target_idx is None:
            return []
        
        # Calculate similarities
        target_vec = self.tfidf_matrix[target_idx]
        similarities = cosine_similarity(target_vec, self.tfidf_matrix).flatten()
        
        # Get top similar products (excluding the target itself)
        similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
        recommendations = [self.products[i] for i in similar_indices if i != target_idx]
        
        return recommendations

# Initialize search system
search_system = ProductSearch(products)

# HTML Templates
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-Commerce Search & Recommendation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .search-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        .search-form {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr auto;
            gap: 15px;
            align-items: end;
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        .form-group label {
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
        }
        input, select, button {
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        .search-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .search-btn:hover {
            transform: translateY(-2px);
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .product-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0,0,0,0.2);
        }
        .product-image {
            width: 100%;
            height: 200px;
            background: #f5f5f5;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            color: #666;
        }
        .product-info {
            padding: 20px;
        }
        .product-name {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }
        .product-category {
            color: #667eea;
            font-size: 0.9rem;
            margin-bottom: 8px;
        }
        .product-price {
            font-size: 1.3rem;
            font-weight: bold;
            color: #2c5530;
            margin-bottom: 8px;
        }
        .product-rating {
            color: #ffd700;
            margin-bottom: 10px;
        }
        .product-description {
            color: #666;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        .recommendations-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 30px;
        }
        .section-title {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #666;
            font-size: 1.1rem;
        }
        .view-details {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
            width: 100%;
            transition: background 0.3s;
        }
        .view-details:hover {
            background: #764ba2;
        }
        @media (max-width: 768px) {
            .search-form {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛍️ E-Commerce Search & Recommendation</h1>
            <p>Find products and get personalized recommendations</p>
        </div>
        
        <div class="search-section">
            <form id="searchForm" class="search-form">
                <div class="form-group">
                    <label for="search">Search Products:</label>
                    <input type="text" id="search" name="search" placeholder="Enter product name, description, or keywords...">
                </div>
                <div class="form-group">
                    <label for="category">Category:</label>
                    <select id="category" name="category">
                        <option value="">All Categories</option>
                        <option value="Electronics">Electronics</option>
                        <option value="Sports">Sports</option>
                        <option value="Home">Home</option>
                        <option value="Fashion">Fashion</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="max_price">Max Price:</label>
                    <input type="number" id="max_price" name="max_price" placeholder="Max price" min="0">
                </div>
                <div class="form-group">
                    <label for="min_rating">Min Rating:</label>
                    <input type="number" id="min_rating" name="min_rating" placeholder="Min rating" min="0" max="5" step="0.1">
                </div>
                <button type="submit" class="search-btn">Search</button>
            </form>
        </div>
        
        <div id="resultsSection">
            {% if products %}
                <h2 class="section-title">Search Results ({{ products|length }} products found)</h2>
                <div class="products-grid">
                    {% for product in products %}
                    <div class="product-card">
                        <div class="product-image">
                            <img src="{{ product.image }}" alt="{{ product.name }}" style="width: 100%; height: 100%; object-fit: cover;">
                        </div>
                        <div class="product-info">
                            <div class="product-name">{{ product.name }}</div>
                            <div class="product-category">{{ product.category }}</div>
                            <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
                            <div class="product-rating">⭐ {{ product.rating }}/5</div>
                            <div class="product-description">{{ product.description }}</div>
                            <button class="view-details" onclick="viewProduct({{ product.id }})">View Details & Recommendations</button>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="no-results">
                    <p>No products found. Try adjusting your search criteria.</p>
                </div>
            {% endif %}
        </div>
        
        {% if recommendations %}
        <div class="recommendations-section">
            <h2 class="section-title">Recommended Products</h2>
            <div class="products-grid">
                {% for product in recommendations %}
                <div class="product-card">
                    <div class="product-image">
                        <img src="{{ product.image }}" alt="{{ product.name }}" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                    <div class="product-info">
                        <div class="product-name">{{ product.name }}</div>
                        <div class="product-category">{{ product.category }}</div>
                        <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
                        <div class="product-rating">⭐ {{ product.rating }}/5</div>
                        <div class="product-description">{{ product.description }}</div>
                        <button class="view-details" onclick="viewProduct({{ product.id }})">View Details</button>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>

    <script>
        function viewProduct(productId) {
            window.location.href = '/product/' + productId;
        }
        
        // Handle form submission
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const params = new URLSearchParams(formData);
            window.location.href = '/search?' + params.toString();
        });
    </script>
</body>
</html>
'''

PRODUCT_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ product.name }} - E-Commerce</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        .back-btn {
            background: white;
            color: #667eea;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin-bottom: 20px;
            transition: background 0.3s;
        }
        .back-btn:hover {
            background: #f5f5f5;
        }
        .product-detail {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            align-items: start;
        }
        .product-image {
            width: 100%;
            height: 400px;
            background: #f5f5f5;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            color: #666;
        }
        .product-info h2 {
            font-size: 2rem;
            color: #333;
            margin-bottom: 10px;
        }
        .product-category {
            color: #667eea;
            font-size: 1.1rem;
            margin-bottom: 15px;
        }
        .product-price {
            font-size: 2rem;
            font-weight: bold;
            color: #2c5530;
            margin-bottom: 15px;
        }
        .product-rating {
            color: #ffd700;
            font-size: 1.2rem;
            margin-bottom: 20px;
        }
        .product-description {
            color: #666;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .buy-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 1.1rem;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .buy-btn:hover {
            transform: translateY(-2px);
        }
        .recommendations-section {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .section-title {
            font-size: 1.5rem;
            color: #333;
            margin-bottom: 20px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .products-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }
        .product-card {
            background: #f9f9f9;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .product-card:hover {
            transform: translateY(-5px);
        }
        .product-card-image {
            width: 100%;
            height: 200px;
            background: #f5f5f5;
        }
        .product-card-info {
            padding: 20px;
        }
        .product-card-name {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }
        .product-card-price {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2c5530;
            margin-bottom: 8px;
        }
        .view-details {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
            width: 100%;
        }
        @media (max-width: 768px) {
            .product-detail {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 10px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛍️ E-Commerce Store</h1>
            <p>Product Details & Recommendations</p>
        </div>
        
        <button class="back-btn" onclick="window.location.href='/'">← Back to Search</button>
        
        <div class="product-detail">
            <div class="product-image">
                <img src="{{ product.image }}" alt="{{ product.name }}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 10px;">
            </div>
            <div class="product-info">
                <h2>{{ product.name }}</h2>
                <div class="product-category">{{ product.category }}</div>
                <div class="product-price">${{ "%.2f"|format(product.price) }}</div>
                <div class="product-rating">⭐ {{ product.rating }}/5</div>
                <div class="product-description">{{ product.description }}</div>
                <button class="buy-btn">Add to Cart 🛒</button>
            </div>
        </div>
        
        {% if recommendations %}
        <div class="recommendations-section">
            <h2 class="section-title">You Might Also Like</h2>
            <div class="products-grid">
                {% for rec_product in recommendations %}
                <div class="product-card">
                    <div class="product-card-image">
                        <img src="{{ rec_product.image }}" alt="{{ rec_product.name }}" style="width: 100%; height: 100%; object-fit: cover;">
                    </div>
                    <div class="product-card-info">
                        <div class="product-card-name">{{ rec_product.name }}</div>
                        <div class="product-category">{{ rec_product.category }}</div>
                        <div class="product-card-price">${{ "%.2f"|format(rec_product.price) }}</div>
                        <div class="product-rating">⭐ {{ rec_product.rating }}/5</div>
                        <button class="view-details" onclick="viewProduct({{ rec_product.id }})">View Details</button>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>

    <script>
        function viewProduct(productId) {
            window.location.href = '/product/' + productId;
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, products=[], recommendations=[])

@app.route('/search')
def search():
    query = request.args.get('search', '')
    category = request.args.get('category', '')
    max_price = request.args.get('max_price', type=float)
    min_rating = request.args.get('min_rating', type=float)
    
    results = search_system.search_products(
        query=query,
        category=category,
        max_price=max_price,
        min_rating=min_rating
    )
    
    return render_template_string(HTML_TEMPLATE, products=results, recommendations=[])

@app.route('/product/<int:product_id>')
def product_detail(product_id):
    product = next((p for p in products if p['id'] == product_id), None)
    if not product:
        return "Product not found", 404
    
    recommendations = search_system.get_recommendations(product_id)
    
    return render_template_string(PRODUCT_TEMPLATE, product=product, recommendations=recommendations)

@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    max_price = request.args.get('max_price', type=float)
    min_rating = request.args.get('min_rating', type=float)
    
    results = search_system.search_products(
        query=query,
        category=category,
        max_price=max_price,
        min_rating=min_rating
    )
    
    return jsonify(results)

@app.route('/api/recommend/<int:product_id>')
def api_recommend(product_id):
    recommendations = search_system.get_recommendations(product_id)
    return jsonify(recommendations)

if __name__ == '__main__':
    print("🚀 Starting E-Commerce Search & Recommendation System...")
    print("📍 Website available at: http://localhost:5000")
    print("🔍 Search API available at: http://localhost:5000/api/search?q=your_query")
    print("💡 Recommendations API available at: http://localhost:5000/api/recommend/1")
    app.run(debug=True, host='0.0.0.0', port=5000)