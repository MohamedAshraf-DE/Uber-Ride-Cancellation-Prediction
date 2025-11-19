🚗 Uber Ride Cancellation Predictor ❌
![Uber Ride Cancellation Predictor](https://github.com/MohamedAshraf-DE/Uber-Ride-Cancellation-Prediction/blob/main/Car-time web app** powered by machine learning to forecast Uber ride cancellations before they happen. Gain insights and actionable predictions to reduce churn, improve trip reliability, and optimize ride-sharing operations for both drivers and riders.

🌟 Why This Project Matters
🏢 For Ride-Hailing Companies & Operations
Anticipate cancellations and reallocate drivers efficiently.

Analyze how time, payment method, and ride details affect cancel rates.

Reduce friction in customer and driver experiences.

👨‍✈️ For Drivers & Riders
Know your ride’s cancellation probability ahead of time.

Understand personal and location-based cancellation trends.

Make smarter booking and acceptance decisions.

💼 Business Value
Minimizes revenue loss from avoidable cancellations.

Highlights top risk factors for operational teams.

Enables data-driven decisions for platform management.

✨ Key Features
Feature	Description
🔮 Live Prediction	Predict cancellation risk instantly for any ride scenario.
🗺️ Location-based Trends	Visualize cancellations by pickup/dropoff zones.
⚡ Time & Payment Analysis	See impact of booking hours, payment methods (cash/card).
📊 Analytics Dashboard	Interactive charts for historical data and prediction outcomes.
🤖 ML Engine	Random Forest classifier trained on real Uber ride and cancellation data.
🖌️ Streamlit UI	Clean, modern dashboard experience for fast exploration and feedback.
🚀 How to Use
Prerequisites

Python 3.9+

Clone the repo & install dependencies:

bash
git clone https://github.com/MohamedAshraf-DE/Uber-Ride-Cancellation-Prediction.git
cd Uber-Ride-Cancellation-Prediction
pip install -r requirements.txt
Model Preparation

The pre-trained cancellation model is included as ride_cancel_model.pkl using Git LFS—no training needed.

Start the App

bash
streamlit run app.py
View predictions in your browser, analyze cancellation risk, and explore trends.

🛠️ Technical Details
Model: RandomForestClassifier (sklearn)

Key Features: Pickup/Dropoff location, driver type, booking time, payment method, ride distance, day of week.

Target: Ride status (Canceled or Completed)

Performance: Metrics (e.g., Accuracy, F1) available in the dashboard.

Tech Stack: pandas, numpy, scikit-learn, streamlit, matplotlib

📈 Demo & Screenshots
Main dashboard visualizes live ride risk, historical patterns, and feature impacts.

Car.jpg used as project cover for clarity.

📞 Connect & Portfolio
🌐 Portfolio: https://mohamed-ashraf-github-io.vercel.app/

🔗 LinkedIn: https://www.linkedin.com/in/mohamed--ashraff

🐙 GitHub: https://github.com/MohamedAshraf-DE/MohamedAshraf.github.io

💼 Freelance Profiles: Upwork | Mostaql | Khamsat | Freelancer | Outlier

Ready to anticipate and reduce Uber ride cancellations with the power of ML? Clone, run, and start exploring the insights today!
