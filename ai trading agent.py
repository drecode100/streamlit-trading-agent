import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 AI Trading Assistant - Excel & Chart Image Analyzer")
st.markdown("---")

# --- Global Variables for ML Model ---
# These are initialized outside any function, making them truly global.
ml_model = None
ml_features = None
ml_target = None
chart_feature_value = 0.0 # Placeholder for image-derived feature
best_hour = "N/A" # Initialize best_hour globally

# ---- SECTION 1: Upload Excel ----
st.header("1. Upload Your Trade Data (Excel)")
uploaded_excel = st.file_uploader("Upload Excel File", type=["xlsx", "xls"], key="excel_uploader")

if uploaded_excel:
    try:
        df = pd.read_excel(uploaded_excel)
        st.subheader("Trade Data Preview")
        st.dataframe(df.head())

        # ---- Basic stats ----
        st.subheader("Basic Trade Stats")
        if 'Profit' in df.columns and 'Symbol' in df.columns:
            st.write(f"**Total Trades:** {len(df)}")
            win_rate = (df['Profit'] > 0).mean() * 100
            st.write(f"**Win Rate:** {round(win_rate, 2)} %")
            
            most_traded_symbol = "N/A"
            if not df['Symbol'].empty:
                most_traded_symbol = df['Symbol'].mode()[0]
            st.write(f"**Most Traded Symbol:** {most_traded_symbol}")
            
            avg_profit = df['Profit'].mean()
            st.write(f"**Average Profit per Trade:** {round(avg_profit, 2)}")

            # Visualize profit/loss
            st.subheader("Profit Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            df['Profit'].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title("Profit/Loss Distribution")
            ax.set_xlabel("Profit/Loss")
            ax.set_ylabel("Number of Trades")
            st.pyplot(fig)

            # Strategy suggestion based on time (if Time column exists)
            if 'Time' in df.columns:
                # Ensure 'Time' is datetime and convert if necessary
                df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
                df.dropna(subset=['Time'], inplace=True) # Drop rows where time conversion failed

                if not df.empty:
                    # Corrected placement for global declaration
                    global best_hour 
                    df['Hour'] = df['Time'].dt.hour
                    hour_win_rate = df.groupby('Hour')['Profit'].apply(lambda x: (x > 0).mean())
                    best_hour = hour_win_rate.idxmax() # Assignment now correctly modifies the global variable
                    st.success(f"✅ **Suggested Best Hour to Trade:** {best_hour}:00 with {round(hour_win_rate.max() * 100, 1)}% win rate")
                else:
                    st.warning("No valid 'Time' data found after parsing.")
            else:
                st.info("💡 Tip: Add a 'Time' column to your Excel for time-based strategy suggestions.")

            # ---- Prepare data for ML model ----
            st.subheader("2. Train AI Model for Trade Prediction")
            st.info("The model will try to predict if a trade will be profitable (Win) or not (Loss).")

            # Feature Engineering (example features)
            ml_df = df.copy()
            ml_df['Is_Win'] = (ml_df['Profit'] > 0).astype(int) # Target variable

            # Add more features if they exist or can be derived
            if 'Volume' in ml_df.columns:
                ml_df['Log_Volume'] = np.log1p(ml_df['Volume'])
            else:
                ml_df['Log_Volume'] = 0 # Dummy if not present

            # Example: previous trade outcome (simple lagging feature)
            ml_df['Prev_Trade_Win'] = ml_df['Is_Win'].shift(1).fillna(0)

            # Example: rolling average profit (if 'Profit' column exists)
            if 'Profit' in ml_df.columns:
                ml_df['Rolling_Avg_Profit'] = ml_df['Profit'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
            else:
                ml_df['Rolling_Avg_Profit'] = 0 # Dummy if not present

            # Use 'Hour' if available, otherwise a dummy or skip
            if 'Hour' in ml_df.columns:
                 features_to_use = ['Hour', 'Log_Volume', 'Prev_Trade_Win', 'Rolling_Avg_Profit']
            else:
                 features_to_use = ['Log_Volume', 'Prev_Trade_Win', 'Rolling_Avg_Profit']
                 st.warning(" 'Time' column not found, 'Hour' feature skipped for ML model.")


            # Filter to only existing columns in features_to_use
            existing_features = [f for f in features_to_use if f in ml_df.columns]
            
            if not existing_features:
                st.error("No suitable features found in Excel data to train the ML model. Please ensure 'Time', 'Profit', or 'Volume' columns are present.")
                global ml_model # Ensure this is reset globally
                ml_model = None
            else:
                X = ml_df[existing_features]
                y = ml_df['Is_Win']

                # Handle potential NaNs introduced by feature engineering
                # For simplicity, we drop rows with NaNs. In a real scenario, consider imputation.
                # Ensure X and y have matching indices after dropping NaNs
                combined_xy = pd.concat([X, y], axis=1).dropna()
                X_clean = combined_xy[existing_features]
                y_clean = combined_xy['Is_Win']

                if X_clean.empty or len(X_clean) < 2: # Need at least 2 samples for split
                    st.warning("Not enough clean data to train the prediction model after feature engineering and NaN removal.")
                    global ml_model
                    ml_model = None
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)

                    # Train the Random Forest Classifier
                    # Using class_weight='balanced' helps with imbalanced datasets
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                    model.fit(X_train, y_train)
                    global ml_model, ml_features, ml_target # Declare global to modify
                    ml_model = model # Store model globally
                    ml_features = existing_features # Store features globally
                    ml_target = y_clean.name # Store target name

                    st.write("Random Forest Model Trained Successfully!")
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"**Model Accuracy (Win/Loss Prediction):** {round(accuracy * 100, 2)}%")
                    st.text("Classification Report:")
                    st.code(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

        else:
            st.warning("Please ensure your Excel file contains 'Profit' and 'Symbol' columns for analysis.")
            global ml_model # Ensure this is reset globally
            ml_model = None
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        global ml_model
        ml_model = None

st.markdown("---")

# ---- SECTION 2: Upload Chart Image ----
st.header("3. Upload a Trading Chart Image for Visual Analysis")
uploaded_img = st.file_uploader("Upload Chart Image", type=["jpg", "jpeg", "png"], key="img_uploader")

if uploaded_img:
    try:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Uploaded Chart", use_column_width=True)
        image_np = np.array(image)

        st.subheader("Chart Analysis (Simplified AI)")
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # Simple edge detection for visual
        edges = cv2.Canny(gray, 50, 150)
        st.image(edges, caption="Detected Edges (Potential Trendlines/Patterns)", use_column_width=True)

        # A very rudimentary "image feature" for ML: proportion of edges
        # In a real scenario, this would be features from a CNN or more complex CV
        global chart_feature_value # Corrected placement for global declaration
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_pixels = np.sum(edges > 0)
        chart_feature_value = edge_pixels / total_pixels # Assignment now correctly modifies the global variable
        st.info(f"**Calculated Chart Edge Density (as a simple feature):** {round(chart_feature_value * 100, 2)}%")
        st.warning("⚠️ **Note:** For true chart pattern recognition (e.g., Head & Shoulders, Flags), you would need to train a specialized Convolutional Neural Network (CNN) model. This 'edge density' is a very basic placeholder feature.")
    except Exception as e:
        st.error(f"Error processing image: {e}")
        global chart_feature_value
        chart_feature_value = 0.0 # Reset on error

st.markdown("---")

# ---- SECTION 3: AI Agent Decision & Signal Generation ----
st.header("4. AI Agent's Trade Signal & Prediction")

if ml_model and ml_features:
    st.subheader("AI Model's Prediction for Your Next Trade")

    # Gather current "market conditions" as input for prediction
    # These would ideally come from live feeds or user inputs
    input_data = {}
    
    st.write("Please provide hypothetical current conditions for a prediction:")
    
    # Dynamically ask for inputs based on features used by the model
    for feature in ml_features:
        if feature == 'Hour':
            input_data['Hour'] = st.slider("Current Trading Hour (0-23)", 0, 23, 9, key="current_hour_input")
        elif feature == 'Log_Volume':
            # Ask for actual volume, then convert to log
            current_volume = st.number_input("Current Trade Volume", value=1000.0, min_value=0.0, key="current_volume_input")
            input_data['Log_Volume'] = np.log1p(current_volume)
        elif feature == 'Prev_Trade_Win':
            input_data['Prev_Trade_Win'] = st.selectbox("Was the Previous Trade a Win?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="prev_trade_win_input")
        elif feature == 'Rolling_Avg_Profit':
            input_data['Rolling_Avg_Profit'] = st.number_input("Average Profit of Last 5 Trades", value=50.0, step=10.0, key="rolling_avg_profit_input")
        # The chart_feature_value is currently only displayed as a factor, not directly fed into *this specific* ML model as a feature.
        # If you add it to ml_features, ensure it's captured here too.


    # Create a DataFrame for prediction
    # Ensure the order of columns matches the training features
    predict_df = pd.DataFrame([input_data], columns=ml_features)

    try:
        # Predict probability of win (class 1)
        win_proba = ml_model.predict_proba(predict_df)[:, 1][0]
        prediction = ml_model.predict(predict_df)[0]

        st.write(f"**Probability of Winning Next Trade:** {round(win_proba * 100, 2)}%")

        signal = "HOLD / Neutral"
        if prediction == 1 and win_proba > 0.65: # High confidence win
            signal = "🟢 **Consider BUY/LONG** (High Win Probability)"
        elif prediction == 0 and win_proba < 0.35: # High confidence loss (implies sell/short or avoid)
            signal = "🔴 **Consider SELL/SHORT or AVOID** (High Loss Probability)"
        elif 0.35 <= win_proba <= 0.65:
            signal = "⚪ **HOLD / Neutral** (Uncertain or Moderate Probability)"

        st.markdown(f"### AI Trade Signal: {signal}")

        st.info(f"**Contributing Factors to Signal:**")
        st.write(f"- Predicted Win Probability: {round(win_proba * 100, 2)}%")
        # Ensure best_hour is displayed, if it was determined
        if best_hour != "N/A":
            st.write(f"- Best Trading Hour identified from your data: **{best_hour}:00**")
        st.write(f"- Current Chart Edge Density (simple visual feature): {round(chart_feature_value * 100, 2)}%")
        
        st.write("*(This signal is based on your historical data and a simplified chart analysis. Always conduct your own research.)*")

    except Exception as e:
        st.error(f"Could not generate prediction. Ensure all required features for the model are available and valid input. Error: {e}")
        st.info(f"Model expected features: {ml_features}")
else:
    st.info("Upload Excel data and train the model in step 2 to enable AI predictions.")

st.markdown("---")

# ---- SECTION 4: Simulated Trade Execution (Conceptual) ----
st.header("5. Simulated Trade Execution (Conceptual)")
st.warning("🛑 **This section is for demonstration of agent actions ONLY and does NOT execute real trades.**")

if ml_model and ml_features:
    st.write("Once a signal is generated, a real AI trading agent would typically take actions like:")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Potential Agent Actions")
        # Initialize signal to avoid NameError if not set in previous block
        signal = "N/A" 
        try:
            # Re-run prediction here to ensure 'signal' is up-to-date based on inputs
            # This is a bit redundant with Section 3, but ensures the buttons respond correctly
            if 'predict_df' in locals() and not predict_df.empty:
                win_proba_for_action = ml_model.predict_proba(predict_df)[:, 1][0]
                prediction_for_action = ml_model.predict(predict_df)[0]
                if prediction_for_action == 1 and win_proba_for_action > 0.65: 
                    signal = "🟢 Consider BUY/LONG"
                elif prediction_for_action == 0 and win_proba_for_action < 0.35:
                    signal = "🔴 Consider SELL/SHORT or AVOID"
                else:
                    signal = "⚪ HOLD / Neutral"
        except Exception as e:
            st.error(f"Error determining signal for simulated actions: {e}")
            signal = "N/A" # Fallback if prediction fails

        if st.button("Simulate BUY Order based on Signal"):
            if "BUY" in signal:
                st.success("✅ **Simulated BUY Order Placed!** (Logic triggered by 'BUY' signal)")
                st.write("Details: Asset X, Quantity Y, Price Z...")
            else:
                st.warning("🚫 No 'BUY' signal detected. Simulated action not taken.")
        
        if st.button("Simulate SELL Order based on Signal"):
            if "SELL" in signal:
                st.error("❌ **Simulated SELL Order Placed!** (Logic triggered by 'SELL' signal)")
                st.write("Details: Asset X, Quantity Y, Price Z...")
            else:
                st.warning("🚫 No 'SELL' signal detected. Simulated action not taken.")

    with col2:
        st.subheader("Agent's Continuous Monitoring")
        st.write("An autonomous agent would constantly:")
        st.markdown("- Monitor real-time market data")
        st.markdown("- Re-evaluate signals based on new information")
        st.markdown("- Manage open positions (stop-loss, take-profit)")
        st.markdown("- Log all decisions and outcomes")

    st.info("For actual automated trading, integration with a broker's API (e.g., MT5, Alpaca, Interactive Brokers) would be required, along with robust risk management and error handling.")

else:
    st.info("Upload Excel data and train the model to enable simulated trade execution features.")

st.markdown("---")
st.markdown("_This AI Trading Assistant is a foundational prototype. For robust automated trading, significant development in real-time data integration, advanced ML models, comprehensive risk management, and secure API connections is essential._")
