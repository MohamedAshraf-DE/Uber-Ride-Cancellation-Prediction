![Uber Ride Cancellation Predictor](https://github.com/MohamedAshraf-DE/Uber-Ride-Cancellation-Prediction/blob/main/Car.jpg)

# 🚗 Uber Ride Cancellation Predictor ❌

A **live web app** that predicts the likelihood of Uber ride cancellations using machine learning.  
This project helps platforms and users understand cancellation risks, improve trip reliability, and optimize operational decisions.

---

## 🌟 Why This Project Matters

### 🏢 For Ride-Hailing Platforms
- **Reduce lost revenue** by anticipating cancellations and reallocating drivers efficiently.
- Simulate “what-if” scenarios such as changing booking times or payment methods.
- Improve rider and driver experience using actionable prediction.

### 🚘 For Drivers & Riders
- See your ride’s cancellation probability before booking or pickup.
- Discover cancellation trends related to trip details, zones, and timing.
- Make smarter decisions and reduce frustration.

### 💼 Business Value
- Enables smarter dispatch and marketing, reducing churn.
- Identifies cancellation causes for targeted platform improvements.
- Powers real-time insights for ride-sharing operations.

---

## ✨ Features & Highlights

| Feature                    | Description                                                      |
|----------------------------|------------------------------------------------------------------|
| 🔮 Live Cancellations      | Get immediate cancellation risk predictions for any ride scenario.|
| 🗺️ Location Analytics      | Visualize cancellations by pickup/dropoff zones.                 |
| ⚡ Time/Payment Insights    | Analyze trends by booking hours and payment method.              |
| 📊 Analytics Dashboard     | Interactive charts and summary statistics.                       |
| 🤖 ML Model                | Random Forest classifier trained on real Uber ride data.         |
| 🎨 Custom Streamlit UI     | Stylish dashboard & smooth navigation.                           |

---

## 🚀 How to Use This App

1. **Prerequisites**
    - Python 3.9+
    - Git

2. **Clone & Install Dependencies**
    ```
    git clone https://github.com/MohamedAshraf-DE/Uber-Ride-Cancellation-Prediction.git
    cd Uber-Ride-Cancellation-Prediction
    pip install -r requirements.txt
    ```

3. **Model Preparation**
    - `ride_cancel_model.pkl` (pre-trained, included via Git LFS)
    - No training needed.

4. **Run the App**
    ```
    streamlit run app.py
    ```
    - Open your browser to explore live predictions and analytics!

---

## 🛠️ Technical Details

- **Model:** RandomForestClassifier (scikit-learn)
- **Features:** Location, time, payment method, ride distance, driver type, weekday/weekend
- **Target:** Completed vs Cancelled rides
- **Libraries:** pandas, numpy, scikit-learn, streamlit, matplotlib

---

## 📞 Contact & Portfolio

- 🌐 Portfolio: [https://mohamed-ashraf-github-io.vercel.app/](https://mohamed-ashraf-github-io.vercel.app/)
- 🔗 LinkedIn: [https://www.linkedin.com/in/mohamed--ashraff](https://www.linkedin.com/in/mohamed--ashraff)
- 🐙 GitHub: [https://github.com/MohamedAshraf-DE/MohamedAshraf.github.io](https://github.com/MohamedAshraf-DE/MohamedAshraf.github.io)
- 💼 Upwork: [Upwork Profile](https://www.upwork.com/freelancers/~0190a07e5b17474f9f?mp_source=share)
- 💼 Mostaql: [Mostaql Profile](https://mostaql.com/u/MohamedA_Data)
- 💼 Khamsat: [Khamsat Profile](https://khamsat.com/user/mohamed_ashraf124)
- 💼 Freelancer: [Freelancer Dashboard](https://www.freelancer.com/dashboard)
- 💼 Outlier: [Outlier Profile](https://app.outlier.ai/profile)

---

**Ready to predict and reduce Uber ride cancellations? Clone, launch, and see it in action!**
