# 🛍️ E-Commerce Product Catalog & Recommendation System

## 📖 Overview
This project is a **smart e-commerce web application** built using **Python (Flask)** and a **modern responsive frontend**.  
It features intelligent product recommendations powered by **data structures** and **algorithms** for an engaging and efficient shopping experience.  

Users can browse products, search and filter by price or category, like their favorite items, and view personalized recommendations — all through a clean, interactive interface.

---

## 🚀 Features
- 🧩 **Category Tree ADT** – Organizes multi-level product categories efficiently.  
- ⚖️ **AVL Tree Indexing** – Enables fast and balanced price-based product searches.  
- 💡 **Recommendation Engine** – Suggests products based on rating, price, category similarity, and recency.  
- 🛒 **Dynamic Frontend Interface** – Clean, responsive design for browsing, filtering, and adding items to cart.  
- 🔗 **Flask Backend APIs** – `/api/products`, `/api/categories`, and `/api/recommend` endpoints for smooth frontend-backend communication.  

---

## 🧠 Tech Stack
| Layer | Technologies Used |
|-------|-------------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, Flask |
| **Algorithms** | Category Tree ADT, AVL Tree, Max Heap |
| **Data Format** | JSON |

---

## 🗂️ Project Structure
Ecommerce-Recommendation/
│
├── ecommerce_full.py # Core logic (data structures & recommendation engine)
├── ecommerce_data.py # Product loading and structure setup
├── server.py # Flask backend and API routes
├── templates/
│ └── index.html # Frontend UI
└── static/
├── style.css # Styling for the frontend
└── script.js # JS for interactivity

yaml
Copy code

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Ecommerce-Recommendation.git
cd Ecommerce-Recommendation
2️⃣ Install Dependencies
bash
Copy code
pip install flask
3️⃣ Run the Server
bash
Copy code
python server.py
4️⃣ Open in Browser
cpp
Copy code
http://127.0.0.1:5000
🎯 How It Works
The backend loads product data and builds a Category Tree and AVL Tree for organization and fast search.

Flask serves the APIs for listing products and generating recommendations.

The frontend fetches data dynamically to render products, filters, and recommendations.

User interactions like “liking” or “adding to cart” update the display instantly.

🧩 Future Enhancements
👤 User authentication (login/signup)

🗄️ Database integration with MongoDB or MySQL

🤖 AI-based recommendation model for more accuracy

📊 Admin dashboard for product management and analytics

💬 Author
Catherin Jenira
🎓 BE CSE | 💡 Data Science & AI Enthusiast
🎧 Podcast Creator – "What If?" by RJ Jeni
📍 Rajalakshmi Institute of Technology

⭐ Acknowledgements
Thanks to my teammates and mentors for their support and collaboration in developing this project.

🌟 Support
If you like this project, give it a star ⭐ on GitHub and share your feedback!

yaml
Copy code

---

Would you like me to make this README **with colorful badges** (like Python, Flask, HTML, CSS, MIT
