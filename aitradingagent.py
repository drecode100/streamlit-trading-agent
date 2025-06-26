import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import plotly.express as px
import plotly.graph_objects as go

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

# --- Custom CSS for a more pleasant look (Basic theming) ---
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #f0f2f6; /* Light gray background */
        color: #333333; /* Darker text */
    }
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {
        color: #004d99; /* Dark blue for headers */
        font-family: 'Segoe UI', sans-serif;
    }
    /* Buttons */
    .stButton>button {
        background-color: #007bff; /* Primary blue */
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    /* Info/Warning/Success boxes */
    div.stAlert {
        border-radius: 8px;
        padding: 15px;
    }
    div.stAlert.info {
        background-color: #e0f7fa;
        color: #00796b;
        border-left: 5px solid #00acc1;
    }
    div.stAlert.warning {
        background-color: #fffde7;
        color: #ef6c00;
        border-left: 5px solid #ffa000;
    }
    div.stAlert.success {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-left: 5px solid #43a047;
    }
    div.stAlert.error {
        background-color: #ffebee;
        color: #c62828;
        border-left: 5px solid #d32f2f;
    }
    /* Sidebar */
    .css-r699ph { /* Target sidebar background */
        background-color: #e6e6e6; /* Slightly darker gray for sidebar */
    }
    .css-1d391kg { /* Target sidebar content padding */
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Session State Variables (for persistence across reruns) ---
if 'ml_model' not in st.session_state:
    st.session_state.ml_model = None
if 'ml_features' not in st.session_state:
    st.session_state.ml_features = None
if 'ml_target' not in st.session_state:
    st.session_state.ml_target = None
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame() # Store the processed DataFrame
if 'chart_feature_value' not in st.session_state:
    st.session_state.chart_feature_value = 0.0
if 'best_hour' not in st.session_state:
    st.session_state.best_hour = "N/A"
if 'current_trade_signal' not in st.session_state:
    st.session_state.current_trade_signal = "No data or signal generated yet."


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio(
    "Go to",
    [
        "Welcome & Overview",
        "1. Data Upload & Performance Analysis",
        "2. AI Model Training",
        "3. Chart Visual Analysis",
        "4. AI Trade Signal",
        "5. Improvement Tips & Actions",
        "6. Simulated Trade Execution"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("This is a prototype AI Trading Assistant. Always perform your own due diligence.")

# --- Add a cache clearing button to the sidebar for easy access ---
if st.sidebar.button("Clear App Cache"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.df = pd.DataFrame() # Clear the main DataFrame
    st.session_state.ml_model = None # Clear trained model
    st.rerun() # Rerun the app to reflect changes
    st.success("Cache cleared! Please re-upload your data.")


# --- Cached Function for Trade Data Processing (Handles multiple CSVs and XLSX) ---
@st.cache_data(show_spinner="Processing trade data...")
def load_and_process_trade_data(uploaded_files):
    """
    Loads multiple CSV/XLSX files into a single DataFrame and performs basic preprocessing.
    If an XLSX file has multiple sheets, all sheets will be read and combined.
    Includes debugging print statements for row counts at each stage.
    """
    all_dfs = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Determine file type and read accordingly
                if uploaded_file.name.endswith('.csv'):
                    df_loaded = pd.read_csv(uploaded_file)
                    all_dfs.append(df_loaded)
                    st.info(f"Loaded {len(df_loaded)} rows from CSV: '{uploaded_file.name}'.")
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    # For Excel files, read all sheets
                    xls = pd.ExcelFile(uploaded_file)
                    for sheet_name in xls.sheet_names:
                        df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                        all_dfs.append(df_sheet)
                        st.info(f"Loaded {len(df_sheet)} rows from sheet '{sheet_name}' in '{uploaded_file.name}'.")
                    st.info(f"Successfully processed all sheets from '{uploaded_file.name}'.")
                else:
                    st.warning(f"Skipping {uploaded_file.name}: Unsupported file type. Please upload CSV or XLSX.")
                    continue
            except Exception as e:
                st.error(f"Error reading {uploaded_file.name}: {e}. Please ensure it's a valid CSV or XLSX file.")
                continue # Skip to the next file
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            st.info(f"Total rows after combining all files/sheets: {len(combined_df)}.")
            
            # --- Robust Data Type Conversions based on provided headers ---
            if 'DATE' in combined_df.columns:
                initial_date_rows = len(combined_df)
                combined_df['DATE'] = pd.to_datetime(combined_df['DATE'], errors='coerce')
                combined_df.dropna(subset=['DATE'], inplace=True)
                st.info(f"Rows after DATE parsing and dropping NaNs: {len(combined_df)} (Lost {initial_date_rows - len(combined_df)} rows).")
                
                # Filter out dates that are unreasonably old (e.g., 1970 epoch start)
                initial_year_filter_rows = len(combined_df)
                combined_df = combined_df[combined_df['DATE'].dt.year >= 2000].copy() 
                st.info(f"Rows after filtering dates before 2000: {len(combined_df)} (Lost {initial_year_filter_rows - len(combined_df)} rows).")

                combined_df.sort_values(by='DATE', inplace=True)
            else:
                st.warning("⚠️ 'DATE' column not found. Some time-based analysis features will be limited and data might not be sorted.")

            numeric_cols = ['LOTS', 'ENTRY', 'EXIT', 'PIPS', 'TP', 'PIPS.1', 'SL', 'PIPS.2', 'PROFIT', 'LOSS', 'DAILY P/L']
            for col in numeric_cols:
                if col in combined_df.columns:
                    # Check for non-numeric values before converting and filling
                    non_numeric_count = pd.to_numeric(combined_df[col], errors='coerce').isna().sum()
                    if non_numeric_count > 0:
                        st.warning(f"Column '{col}' had {non_numeric_count} non-numeric values converted to 0.")
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
            
            if 'PROFIT' in combined_df.columns and 'LOSS' in combined_df.columns:
                combined_df['Net_Profit'] = combined_df['PROFIT'] - combined_df['LOSS']
            elif 'PROFIT' in combined_df.columns:
                combined_df['Net_Profit'] = combined_df['PROFIT']
            elif 'DAILY P/L' in combined_df.columns:
                combined_df['Net_Profit'] = pd.to_numeric(combined_df['DAILY P/L'], errors='coerce').fillna(0)
            else:
                st.error("❌ Critical: 'PROFIT' or 'DAILY P/L' columns not found. Cannot calculate trade profitability. Returning empty DataFrame.")
                return pd.DataFrame()

            if 'Net_Profit' in combined_df.columns:
                initial_net_profit_rows = len(combined_df)
                combined_df['Net_Profit'] = pd.to_numeric(combined_df['Net_Profit'], errors='coerce').fillna(0)
                # No rows are dropped here, just NaNs filled, so no specific row count change
                
                combined_df['Rolling_Avg_Profit'] = combined_df['Net_Profit'].rolling(window=10, min_periods=1).mean()
                combined_df['Is_Win_Bool'] = (combined_df['Net_Profit'] > 0)
                combined_df['Rolling_Avg_Win_Rate'] = combined_df['Is_Win_Bool'].rolling(window=10, min_periods=1).mean() * 100
                combined_df.drop(columns=['Is_Win_Bool'], inplace=True)
            
            st.success(f"Final processed DataFrame contains {len(combined_df)} rows for analysis.")
            return combined_df
    st.warning("No files uploaded or no valid data could be processed.")
    return pd.DataFrame()

# --- Cached Function for ML Model Training ---
@st.cache_resource(show_spinner="Training AI model...")
def train_ml_model(df_trainable):
    """Trains a RandomForestClassifier model and returns it with features and target."""
    if df_trainable.empty or 'Net_Profit' not in df_trainable.columns:
        if df_trainable.empty:
            st.error("Cannot train model: Data is empty.")
        else:
            st.error("Cannot train model: 'Net_Profit' column not found.")
        return None, None, None

    ml_df = df_trainable.copy()
    ml_df['Is_Win'] = (ml_df['Net_Profit'] > 0).astype(int)

    # Ensure feature columns exist before attempting to create new ones based on them
    if 'LOTS' in ml_df.columns:
        ml_df['Log_LOTS'] = np.log1p(ml_df['LOTS'])
    else:
        ml_df['Log_LOTS'] = 0 # Default to 0 if LOTS is missing

    ml_df['Prev_Trade_Win'] = ml_df['Is_Win'].shift(1).fillna(0)

    if 'Rolling_Avg_Profit' not in ml_df.columns:
        if 'Net_Profit' in ml_df.columns:
            ml_df['Rolling_Avg_Profit'] = ml_df['Net_Profit'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
        else:
            ml_df['Rolling_Avg_Profit'] = 0 # Default to 0 if Net_Profit is also missing
    else:
        ml_df['Rolling_Avg_Profit'] = ml_df['Rolling_Avg_Profit'].shift(1).fillna(0)

    if 'DATE' in ml_df.columns:
        ml_df['Hour'] = ml_df['DATE'].dt.hour
        ml_df['DayOfWeek'] = ml_df['DATE'].dt.dayofweek
    else:
        ml_df['Hour'] = 0 # Default to 0 if DATE is missing
        ml_df['DayOfWeek'] = 0 # Default to 0 if DATE is missing

    features_to_use = []
    if 'Hour' in ml_df.columns: features_to_use.append('Hour')
    if 'DayOfWeek' in ml_df.columns: features_to_use.append('DayOfWeek')
    if 'Log_LOTS' in ml_df.columns: features_to_use.append('Log_LOTS')
    features_to_use.extend(['Prev_Trade_Win', 'Rolling_Avg_Profit'])

    existing_features = [f for f in features_to_use if f in ml_df.columns]

    if not existing_features:
        st.error("No suitable features found in your data to train the ML model. Please ensure columns like 'DATE', 'LOTS', 'Net_Profit' are present and correctly processed.")
        return None, None, None
    
    X = ml_df[existing_features]
    y = ml_df['Is_Win']

    # Drop any remaining NaNs in features or target before splitting
    combined_xy = pd.concat([X, y], axis=1).dropna()
    X_clean = combined_xy[existing_features]
    y_clean = combined_xy['Is_Win']

    if X_clean.empty or len(X_clean) < 2:
        st.warning(f"Not enough clean data to train the prediction model. Only {len(X_clean)} rows remaining after cleaning.")
        return None, None, None
    
    # Ensure all feature columns are numeric, filling any remaining NaNs with 0 (should be minimal if previous steps are robust)
    X_clean = X_clean.apply(pd.to_numeric, errors='coerce').fillna(0)

    try:
        if len(y_clean.unique()) < 2:
            st.warning("Only one class (all wins or all losses) found in data. Cannot train classification model. Need both wins and losses.")
            return None, None, None

        # Stratify ensures train/test sets have similar proportions of win/loss
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean)

        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.success("Random Forest Model Trained Successfully!")
        st.write(f"**Model Accuracy (Win/Loss Prediction):** {round(accuracy * 100, 2)}%")
        with st.expander("View Classification Report"):
            st.code(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

        st.subheader("Feature Importance for Win/Loss Prediction")
        feature_importances = pd.Series(model.feature_importances_, index=existing_features).sort_values(ascending=False)
        fig_feat_imp = px.bar(feature_importances, 
                              x=feature_importances.index, y=feature_importances.values,
                              title="Impact of Features on Trade Win/Loss Prediction",
                              labels={'x': 'Feature', 'y': 'Importance'},
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_feat_imp.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_feat_imp, use_container_width=True)

        return model, existing_features, y_clean.name
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.info("Check if your data has enough variety (both wins and losses) and if essential columns are present and correctly formatted.")
        return None, None, None


# --- Page Content based on Navigation ---

# --- Welcome & Overview ---
if page_selection == "Welcome & Overview":
    st.title("Welcome to the AI Trading Assistant!")
    st.markdown("""
    This application is designed to help you analyze your trading performance, identify key patterns, and receive AI-driven insights.
    
    **How to use:**
    1.  **Upload Your Trade Data:** Start by uploading your historical trade data in CSV or XLSX format. If your Excel file has multiple sheets, all will be processed!
    2.  **Explore Performance:** The app will visualize your equity curve, profit distribution, and rolling performance.
    3.  **Train AI Model:** An AI model will be trained to predict win/loss outcomes based on your historical trades.
    4.  **Get AI Signals:** Input current trade conditions and a chart image to receive a hypothetical trade signal.
    5.  **Review Tips:** Get personalized improvement tips based on your data and the AI model.
    """)
    st.info("Navigate through the sections using the sidebar on the left.")

# ---- SECTION 1: Upload Trade Data & Performance Analysis ----
elif page_selection == "1. Data Upload & Performance Analysis":
    st.header("1. Upload Your Trade Data & Performance Analysis")
    st.markdown("Upload your trade history files (CSV or XLSX). You can upload multiple files, and if an Excel file has multiple sheets, **all sheets will be analyzed**.")
    uploaded_csv_files = st.file_uploader("Upload Trade File(s)", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="csv_uploader")

    if uploaded_csv_files:
        st.session_state.df = load_and_process_trade_data(uploaded_csv_files)
        
        if not st.session_state.df.empty:
            df = st.session_state.df # Use a local variable for convenience

            st.subheader("📊 Trade Data Preview")
            st.dataframe(df.head())

            st.markdown("---")
            st.subheader("📈 Detailed Performance Metrics")
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)
                if 'Net_Profit' in df.columns:
                    total_net_profit = df['Net_Profit'].sum()
                    total_trades = len(df)
                    winning_trades = df[df['Net_Profit'] > 0]
                    losing_trades = df[df['Net_Profit'] <= 0]

                    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
                    avg_win = winning_trades['Net_Profit'].mean() if len(winning_trades) > 0 else 0
                    avg_loss = losing_trades['Net_Profit'].mean() if len(losing_trades) > 0 else 0

                    total_gross_profit = winning_trades['Net_Profit'].sum()
                    total_gross_loss = abs(losing_trades['Net_Profit'].sum())
                    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf

                    with col1:
                        st.metric(label="Total Net P/L", value=f"${total_net_profit:,.2f}")
                        st.metric(label="Win Rate", value=f"{win_rate:,.2f}%")
                    with col2:
                        st.metric(label="Total Trades", value=f"{total_trades}")
                        st.metric(label="Profit Factor", value=f"{profit_factor:,.2f}")
                    with col3:
                        st.metric(label="Avg Winning Trade", value=f"${avg_win:,.2f}")
                        st.metric(label="Avg Losing Trade", value=f"${avg_loss:,.2f}")
                    
                    if 'DATE' in df.columns:
                        df['Cumulative_Profit'] = df['Net_Profit'].cumsum()
                        df['Peak'] = df['Cumulative_Profit'].expanding(min_periods=1).max()
                        df['Drawdown'] = df['Cumulative_Profit'] - df['Peak']
                        max_drawdown = df['Drawdown'].min() if not df['Drawdown'].empty else 0
                        st.metric(label="Maximum Drawdown", value=f"${max_drawdown:,.2f}")
                    else:
                        st.warning("⚠️ 'DATE' column is needed to calculate Max Drawdown.")

            st.markdown("---")
            with st.expander("View Performance Charts", expanded=True):
                if 'DATE' in df.columns and 'Cumulative_Profit' in df.columns:
                    st.subheader("Equity Curve")
                    fig_equity = px.line(df, x='DATE', y='Cumulative_Profit', 
                                         title="Cumulative Profit Over Time",
                                         labels={'DATE': 'Date', 'Cumulative_Profit': 'Cumulative Profit ($)'})
                    st.plotly_chart(fig_equity, use_container_width=True)

                st.subheader("Profit Distribution")
                fig_profit = px.histogram(df, x='Net_Profit', nbins=30, 
                                         title="Net Profit/Loss Distribution",
                                         labels={'Net_Profit': 'Net Profit/Loss'},
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_profit, use_container_width=True)

                if 'DATE' in df.columns and 'Rolling_Avg_Profit' in df.columns and 'Rolling_Avg_Win_Rate' in df.columns:
                    st.subheader("📊 Rolling Performance Over Time")
                    
                    fig_rolling_profit = px.line(df, x='DATE', y='Rolling_Avg_Profit',
                                                 title="Rolling Average Profit (Last 10 Trades)",
                                                 labels={'DATE': 'Date', 'Rolling_Avg_Profit': 'Avg Profit ($)'})
                    st.plotly_chart(fig_rolling_profit, use_container_width=True)

                    fig_rolling_win_rate = px.line(df, x='DATE', y='Rolling_Avg_Win_Rate',
                                                   title="Rolling Win Rate (Last 10 Trades)",
                                                   labels={'DATE': 'Date', 'Rolling_Avg_Win_Rate': 'Win Rate (%)'})
                    st.plotly_chart(fig_rolling_win_rate, use_container_width=True)

                    current_rolling_profit = df['Rolling_Avg_Profit'].iloc[-1]
                    current_rolling_win_rate = df['Rolling_Avg_Win_Rate'].iloc[-1]
                    overall_avg_profit = df['Net_Profit'].mean()
                    overall_win_rate = (df['Net_Profit'] > 0).mean() * 100

                    st.markdown("---")
                    st.subheader("Insights from Rolling Performance:")
                    if current_rolling_profit > overall_avg_profit * 1.1 and current_rolling_win_rate > overall_win_rate * 1.1:
                        st.success("🔥 **Hot Streak!** Your recent performance is significantly better than your overall average. Maintain discipline but consider if conditions are exceptionally favorable.")
                    elif current_rolling_profit < overall_avg_profit * 0.9 and current_rolling_win_rate < overall_win_rate * 0.9:
                        st.warning("❄️ **Cold Streak?** Your recent performance is below your overall average. It might be a good time to review your strategy, reduce position sizes, or consider taking a short break.")
                    else:
                        st.info("⚖️ Your recent performance is consistent with your overall average. Continue monitoring and adhering to your strategy.")
                    st.markdown("---")

                if 'PAIR' in df.columns:
                    st.subheader("Profitability by PAIR")
                    pair_profit = df.groupby('PAIR')['Net_Profit'].sum().sort_values(ascending=False)
                    fig_pair_profit = px.bar(pair_profit, x=pair_profit.index, y=pair_profit.values,
                                            title="Total Net Profit by PAIR",
                                            labels={'x': 'PAIR', 'y': 'Total Net Profit ($)'},
                                            color=pair_profit.values,
                                            color_continuous_scale=px.colors.sequential.Viridis)
                    st.plotly_chart(fig_pair_profit, use_container_width=True)

                if 'ORDER' in df.columns:
                    st.subheader("Performance by Order Type (BUY/SELL)")
                    order_performance = df.groupby('ORDER')['Net_Profit'].agg(['sum', 'mean', lambda x: (x > 0).mean() * 100])
                    order_performance.columns = ['Total Profit', 'Avg Profit', 'Win Rate %']
                    st.dataframe(order_performance.round(2))

                if 'DATE' in df.columns:
                    df['Hour'] = df['DATE'].dt.hour
                    hour_win_rate = df.groupby('Hour')['Net_Profit'].apply(lambda x: (x > 0).mean())
                    if not hour_win_rate.empty:
                        st.session_state.best_hour = hour_win_rate.idxmax() 
                        st.success(f"✅ **Suggested Best Hour to Trade:** {st.session_state.best_hour}:00 with {round(hour_win_rate.max() * 100, 1)}% win rate (based on Net Profit)")
                        fig_hourly_win = px.bar(hour_win_rate, x=hour_win_rate.index, y=hour_win_rate.values * 100,
                                                title="Win Rate by Hour of Day",
                                                labels={'x': 'Hour of Day (24-hour)', 'y': 'Win Rate (%)'},
                                                color=hour_win_rate.values * 100,
                                                color_continuous_scale=px.colors.sequential.Plasma)
                        st.plotly_chart(fig_hourly_win, use_container_width=True)
                    else:
                        st.session_state.best_hour = "N/A"
                        st.warning("Could not calculate best hour from 'DATE' data.")
                else:
                    st.session_state.best_hour = "N/A" 
                    st.info("💡 Tip: Ensure your 'DATE' column is correctly formatted for time-based strategy suggestions.")
            
        else:
            st.warning("Please upload your trade data to see the analysis.")

    else: # If no files are uploaded initially, reset df
        st.session_state.df = pd.DataFrame()


# ---- SECTION 2: AI Model Training ----
elif page_selection == "2. AI Model Training":
    st.header("2. Train AI Model for Trade Prediction")
    st.info("The model learns from your historical data to predict if a trade will be a win or loss.")
    
    if not st.session_state.df.empty:
        if st.button("Train Prediction Model", key="train_model_button"):
            st.session_state.ml_model, st.session_state.ml_features, st.session_state.ml_target = train_ml_model(st.session_state.df)
        
        # Display model info if already trained
        if st.session_state.ml_model:
            st.subheader("Current Model Status:")
            st.success("Model is trained and ready for predictions!")
            if st.session_state.ml_features:
                st.write(f"**Features used:** {', '.join(st.session_state.ml_features)}")
    else:
        st.info("Please upload your trade data in the '1. Data Upload & Performance Analysis' section first.")


# ---- SECTION 3: Upload Chart Image ----
elif page_selection == "3. Chart Visual Analysis":
    st.header("3. Upload a Trading Chart Image for Visual Analysis")
    st.markdown("Upload a chart image to get a basic 'edge density' analysis, which is a placeholder for more advanced chart pattern recognition.")
    uploaded_img = st.file_uploader("Upload Chart Image", type=["jpg", "jpeg", "png"], key="img_uploader")

    if uploaded_img:
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Uploaded Chart", use_column_width=True)
            image_np = np.array(image)

            st.subheader("Chart Analysis (Simplified AI)")
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            edges = cv2.Canny(gray, 50, 150)
            st.image(edges, caption="Detected Edges (Potential Trendlines/Patterns)", use_column_width=True)

            total_pixels = edges.shape[0] * edges.shape[1]
            edge_pixels = np.sum(edges > 0)
            st.session_state.chart_feature_value = edge_pixels / total_pixels 
            st.info(f"**Calculated Chart Edge Density (as a simple feature):** {round(st.session_state.chart_feature_value * 100, 2)}%")
            st.warning("⚠️ **Note:** For true chart pattern recognition (e.g., Head & Shoulders, Flags), you would need to train a specialized Convolutional Neural Network (CNN) model. This 'edge density' is a very basic placeholder feature.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.session_state.chart_feature_value = 0.0 
    else:
        st.info("Upload a chart image to analyze its edge density.")

# ---- SECTION 4: AI Agent Decision & Signal Generation ----
elif page_selection == "4. AI Trade Signal":
    st.header("4. AI Agent's Trade Signal & Prediction")
    st.markdown("Receive a hypothetical trade signal based on your trained AI model and current market conditions.")

    if st.session_state.ml_model and st.session_state.ml_features:
        st.subheader("Provide Current Trade Conditions:")

        input_data = {}
        for feature in st.session_state.ml_features:
            if feature == 'Hour':
                input_data['Hour'] = st.slider("Current Trading Hour (0-23)", 0, 23, 9, key="pred_hour_input")
            elif feature == 'DayOfWeek':
                input_data['DayOfWeek'] = st.selectbox("Current Day of Week", 
                                                        options=list(range(7)), 
                                                        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x],
                                                        key="pred_dayofweek_input")
            elif feature == 'Log_LOTS':
                current_lots = st.number_input("Current Trade LOTS", value=0.1, min_value=0.0, key="pred_lots_input")
                input_data['Log_LOTS'] = np.log1p(current_lots)
            elif feature == 'Prev_Trade_Win':
                input_data['Prev_Trade_Win'] = st.selectbox("Was the Previous Trade a Win?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", key="pred_prev_win_input")
            elif feature == 'Rolling_Avg_Profit':
                input_data['Rolling_Avg_Profit'] = st.number_input("Average Profit of Last 5 Trades", value=50.0, step=10.0, key="pred_rolling_avg_input")

        predict_df = pd.DataFrame([input_data], columns=st.session_state.ml_features)

        try:
            predict_df = predict_df.apply(pd.to_numeric, errors='coerce').fillna(0)

            win_proba = st.session_state.ml_model.predict_proba(predict_df)[:, 1][0]
            prediction = st.session_state.ml_model.predict(predict_df)[0]

            st.write(f"**Probability of Winning Next Trade:** {round(win_proba * 100, 2)}%")

            if prediction == 1 and win_proba > 0.65:
                st.session_state.current_trade_signal = "🟢 **Consider BUY/LONG** (High Win Probability)"
            elif prediction == 0 and win_proba < 0.35:
                st.session_state.current_trade_signal = "🔴 **Consider SELL/SHORT or AVOID** (High Loss Probability)"
            else:
                st.session_state.current_trade_signal = "⚪ **HOLD / Neutral** (Uncertain or Moderate Probability)"

            st.markdown(f"### AI Trade Signal: {st.session_state.current_trade_signal}")

            st.info(f"**Contributing Factors to Signal:**")
            st.write(f"- Predicted Win Probability: {round(win_proba * 100, 2)}%")
            
            if st.session_state.best_hour != "N/A":
                st.write(f"- Best Trading Hour identified from your data: **{st.session_state.best_hour}:00**")
            else:
                 st.write("- Best trading hour not available (requires 'DATE' column in data).")

            if st.session_state.chart_feature_value != 0.0:
                st.write(f"- Current Chart Edge Density (simple visual feature): {round(st.session_state.chart_feature_value * 100, 2)}%")
            else:
                st.write("- Chart analysis not performed (upload image in Section 3).")
            
            st.write("*(This signal is based on your historical data and a simplified chart analysis. Always conduct your own research.)*")

        except Exception as e:
            st.error(f"Could not generate prediction. Ensure all required features for the model are available and valid input. Error: {e}")
            st.info(f"Model expected features: {st.session_state.ml_features}")
            st.session_state.current_trade_signal = "Error: Could not generate signal." 
    else:
        st.info("Please upload CSV/XLSX data in '1. Data Upload' and train the model in '2. AI Model Training' to enable AI predictions.")

# ---- SECTION 5: Corrective Actions & Improvement Tips ----
elif page_selection == "5. Improvement Tips & Actions":
    st.header("5. Corrective Actions & Improvement Tips")
    st.markdown("Receive personalized insights and actionable tips based on your trading performance and the AI model's analysis.")

    if not st.session_state.df.empty and 'Net_Profit' in st.session_state.df.columns:
        df = st.session_state.df # Use a local variable for convenience
        st.subheader("Data-Driven Insights:")

        if 'DATE' in df.columns:
            df['DayOfWeek'] = df['DATE'].dt.dayofweek
            day_of_week_profit = df.groupby('DayOfWeek')['Net_Profit'].sum()
            if not day_of_week_profit.empty:
                best_day_of_week_num = day_of_week_profit.idxmax()
                worst_day_of_week_num = day_of_week_profit.idxmin()
                days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                st.info(f"💡 **Trading Day Performance:** You tend to perform best on **{days_map.get(best_day_of_week_num, 'N/A')}** and worst on **{days_map.get(worst_day_of_week_num, 'N/A')}**.")
                if day_of_week_profit[worst_day_of_week_num] < 0:
                    st.warning(f"⚠️ **Corrective Action:** Consider reducing trading activity or reviewing your strategy specifically on {days_map.get(worst_day_of_week_num, 'N/A')}s, as it's been a net loss day.")
            
            st.info("📊 **Tip for Trade Duration:** If you track exact entry and exit times, we could analyze if shorter or longer trades are more profitable for you, leading to optimization of holding periods.")

        # Analysis based on ML model insights
        if st.session_state.ml_model and st.session_state.ml_features:
            st.subheader("Insights from Prediction Model:")
            feature_importances = pd.Series(st.session_state.ml_model.feature_importances_, index=st.session_state.ml_features).sort_values(ascending=False)
            top_feature = feature_importances.index[0]
            st.write(f"- Your prediction model suggests that **'{top_feature}'** is the most influential factor in determining if your trades will be a win or loss.")
            st.info(f"🔍 **Corrective Action:** Pay extra attention to '{top_feature}' when entering new trades. Try to optimize conditions around this factor.")
            
            if 'LOTS' in df.columns and 'Net_Profit' in df.columns: # Check if columns exist in the original df
                try:
                    # Calculate correlation only if both columns are numeric and non-zero variance
                    if df['LOTS'].std() > 0 and df['Net_Profit'].std() > 0:
                        if df['Net_Profit'].corr(df['LOTS']) < 0:
                            st.warning("⚠️ **Corrective Action:** It appears larger 'LOTS' sizes might be negatively correlated with your profitability. Consider reviewing your position sizing strategy.")
                        else:
                            st.info("✅ Your position sizing ('LOTS') seems to be generally aligned with profitability trends.")
                except Exception as e:
                    st.info(f"Could not analyze LOTS vs Profit correlation: {e}")

        else:
            st.info("Train the AI model in '2. AI Model Training' to get model-driven tips.")
    else:
        st.info("Upload and process your trade data in '1. Data Upload' to generate detailed analysis and corrective actions.")

# ---- SECTION 6: Simulated Trade Execution (Conceptual) ----
elif page_selection == "6. Simulated Trade Execution":
    st.header("6. Simulated Trade Execution (Conceptual)")
    st.warning("🛑 **This section is for demonstration of agent actions ONLY and does NOT execute real trades.**")

    if st.session_state.ml_model and st.session_state.current_trade_signal != "No data or signal generated yet." and "Error" not in st.session_state.current_trade_signal: 
        st.write("Once a signal is generated, a real AI trading agent would typically take actions like:")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Potential Agent Actions")
            
            if st.button("Simulate BUY Order based on Signal", key="sim_buy_button"):
                if "BUY" in st.session_state.current_trade_signal: 
                    st.success("✅ **Simulated BUY Order Placed!** (Logic triggered by 'BUY' signal)")
                    st.write("Details: Asset X, Quantity Y, Price Z...")
                else:
                    st.warning("🚫 No 'BUY' signal detected. Simulated action not taken.")
            
            if st.button("Simulate SELL Order based on Signal", key="sim_sell_button"):
                if "SELL" in st.session_state.current_trade_signal: 
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
        st.info("Upload CSV/XLSX data, train the model, and generate a signal in the previous steps to enable simulated trade execution features.")

st.markdown("---")
st.markdown("_This AI Trading Assistant is a foundational prototype. For robust automated trading, significant development in real-time data integration, advanced ML models, comprehensive risk management, and secure API connections is essential._")
