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
import time # For simulated loading

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Trading Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for an elegant, professional, and ergonomic look ---
# Simplified for TokenError fix by removing some internal comments and consolidating lines.
st.markdown("""
<style>
    /* Global Font: iPhone-like (San Francisco equivalent) */
    html, body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
        color: #333d47; /* Darker text for better contrast */
    }

    /* Overall App Background */
    .stApp { background: #f0f2f5; }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #004d99; font-weight: 700; margin-top: 1.5rem; margin-bottom: 0.8rem;
    }
    h1 { font-size: 2.5rem; }
    h2 { font-size: 2rem; }
    h3 { font-size: 1.6rem; }

    /* Landing Page Specific Styles (Welcome & Overview) */
    .welcome-section {
        background: linear-gradient(135deg, #e0f2f7 0%, #c1e4f4 100%);
        padding: 3rem 2rem; border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 77, 153, 0.1);
        text-align: center; margin-bottom: 2rem;
    }
    .welcome-section h1 {
        color: #003366; font-size: 3.5rem; margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
    }
    .welcome-section p {
        color: #0056b3; font-size: 1.2rem; line-height: 1.6;
        max-width: 800px; margin: 0.5rem auto 1.5rem auto;
    }

    /* General containers/cards */
    .stContainer, .streamlit-expander {
        background-color: #ffffff; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        padding: 1.5rem; margin-bottom: 1.5rem;
        border: 1px solid #e0e6ec;
    }

    /* Buttons - more modern look */
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 8px; border: none;
        padding: 0.7rem 1.4rem; font-size: 1rem; font-weight: 600;
        transition: all 0.2s ease-in-out; cursor: pointer;
        box-shadow: 0 2px 5px rgba(0, 123, 255, 0.2);
    }
    .stButton>button:hover {
        background-color: #0056b3; transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.3);
    }
    .stButton>button:active {
        background-color: #004085; transform: translateY(0);
        box-shadow: 0 1px 3px rgba(0, 123, 255, 0.2);
    }

    /* Input widgets */
    .stTextInput>div>div>input, .stNumberInput>div>div>input,
    .stSelectbox>div>div>select, .stSlider>div>div>div>div {
        border-radius: 8px; border: 1px solid #dcdfe6;
        padding: 0.6rem 0.9rem; font-size: 1rem; color: #495057;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #66b3ff; box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
    }

    /* Metrics - professional and distinct */
    [data-testid="stMetric"] {
        background-color: #f7f9fb; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;
        text-align: center; border: 1px solid #e9ecef;
    }
    [data-testid="stMetric"] label {
        font-size: 1rem; color: #6c757d; font-weight: 500; margin-bottom: 0.5rem;
    }
    [data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 2.2rem; font-weight: 700; color: #0056b3;
    }
    [data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;
    }

    /* Info/Warning/Success/Error boxes - vibrant and clear */
    div.stAlert {
        border-radius: 8px; padding: 1rem 1.25rem; margin-bottom: 1rem;
        font-size: 0.95rem; line-height: 1.4; border: none;
    }
    div.stAlert.info { background-color: #e3f2fd; color: #1a73e8; border-left: 6px solid #2196f3; }
    div.stAlert.warning { background-color: #fff3e0; color: #f57c00; border-left: 6px solid #fb8c00; }
    div.stAlert.success { background-color: #e8f5e9; color: #388e3c; border-left: 6px solid #4caf50; }
    div.stAlert.error { background-color: #ffebee; color: #d32f2f; border-left: 6px solid #ef5350; }

    /* Sidebar styling */
    .st-emotion-cache-r699ph { /* Target sidebar background */
        background-color: #ffffff; padding-top: 2rem; padding-left: 1.5rem;
        padding-right: 1.5rem; box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    .st-emotion-cache-1d391kg { /* Target sidebar content padding */ padding-top: 2rem; }
    .st-emotion-cache-r699ph .stRadio>label {
        font-weight: 600; color: #495057; padding: 0.6rem 0.2rem; transition: color 0.2s ease;
    }
    .st-emotion-cache-r699ph .stRadio>label:hover { color: #007bff; }
    .st-emotion-cache-r699ph .stRadio [data-testid="stCheckableInput"]:checked + div {
        background-color: #e6f2ff; border-radius: 6px; color: #007bff;
    }

    /* Plotly chart container */
    .js-plotly-plot {
        border-radius: 10px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        overflow: hidden; margin-bottom: 1.5rem;
        background-color: #ffffff; border: 1px solid #e0e6ec;
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
if 'simulated_trade_log' not in st.session_state:
    st.session_state.simulated_trade_log = []


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
                st.error(f"Error reading {uploaded_file.name}: {e}. Please ensure it's a valid CSV or XLSX file and not corrupted.")
                continue # Skip to the next file
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            initial_combined_rows = len(combined_df)
            st.info(f"Total rows after combining all files/sheets: {initial_combined_rows}.")
            
            # --- Robust Data Type Conversions based on provided headers ---
            # DATE column processing
            if 'DATE' in combined_df.columns:
                initial_date_rows = len(combined_df)
                combined_df['DATE'] = pd.to_datetime(combined_df['DATE'], errors='coerce')
                
                # Drop rows where DATE conversion failed
                combined_df.dropna(subset=['DATE'], inplace=True)
                if len(combined_df) < initial_date_rows:
                    st.warning(f"Warning: {initial_date_rows - len(combined_df)} rows dropped due to invalid 'DATE' format.")
                st.info(f"Rows after 'DATE' parsing and dropping NaNs: {len(combined_df)}.")
                
                # Filter out dates that are unreasonably old (e.g., 1970 epoch start)
                initial_year_filter_rows = len(combined_df)
                combined_df = combined_df[combined_df['DATE'].dt.year >= 2000].copy() 
                if len(combined_df) < initial_year_filter_rows:
                    st.warning(f"Warning: {initial_year_filter_rows - len(combined_df)} rows dropped because 'DATE' was before year 2000.")
                st.info(f"Rows after filtering dates before 2000: {len(combined_df)}.")

                combined_df.sort_values(by='DATE', inplace=True)
            else:
                st.warning("⚠️ 'DATE' column not found. Some time-based analysis features will be limited and data might not be sorted. Please ensure your data includes a 'DATE' column for full functionality.")

            # Numeric columns conversion
            numeric_cols = ['LOTS', 'ENTRY', 'EXIT', 'PIPS', 'TP', 'PIPS.1', 'SL', 'PIPS.2', 'PROFIT', 'LOSS', 'DAILY P/L']
            for col in numeric_cols:
                if col in combined_df.columns:
                    # Check for non-numeric values before converting and filling
                    non_numeric_count = pd.to_numeric(combined_df[col], errors='coerce').isna().sum()
                    if non_numeric_count > 0:
                        st.warning(f"Column '{col}' had {non_numeric_count} non-numeric values. These were converted to 0.")
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce').fillna(0)
            
            # Net_Profit calculation - critical for analysis
            net_profit_calculated = False
            if 'PROFIT' in combined_df.columns and 'LOSS' in combined_df.columns:
                combined_df['Net_Profit'] = combined_df['PROFIT'] - combined_df['LOSS']
                net_profit_calculated = True
            elif 'PROFIT' in combined_df.columns:
                combined_df['Net_Profit'] = combined_df['PROFIT']
                net_profit_calculated = True
            elif 'DAILY P/L' in combined_df.columns:
                combined_df['Net_Profit'] = pd.to_numeric(combined_df['DAILY P/L'], errors='coerce').fillna(0)
                net_profit_calculated = True

            if not net_profit_calculated:
                st.error("❌ Critical: 'PROFIT', 'LOSS', or 'DAILY P/L' columns not found after processing. Cannot calculate trade profitability. Returning empty DataFrame.")
                return pd.DataFrame() # Return empty if crucial columns are missing
            
            # Ensure Net_Profit is numeric and fill any remaining NaNs
            if 'Net_Profit' in combined_df.columns:
                combined_df['Net_Profit'] = pd.to_numeric(combined_df['Net_Profit'], errors='coerce').fillna(0)
                
                # Calculate rolling metrics only if Net_Profit is available and not empty
                if not combined_df['Net_Profit'].empty:
                    combined_df['Is_Win_Bool'] = (combined_df['Net_Profit'] > 0)
                    combined_df['Rolling_Avg_Profit'] = combined_df['Net_Profit'].rolling(window=10, min_periods=1).mean()
                    combined_df['Rolling_Avg_Win_Rate'] = combined_df['Is_Win_Bool'].rolling(window=10, min_periods=1).mean() * 100
                    combined_df.drop(columns=['Is_Win_Bool'], inplace=True, errors='ignore') # errors='ignore' to prevent error if column already dropped
                else:
                    st.warning("No data in 'Net_Profit' column to calculate rolling averages.")
                    combined_df['Rolling_Avg_Profit'] = 0
                    combined_df['Rolling_Avg_Win_Rate'] = 0
            else:
                st.error("Error: 'Net_Profit' column not created. This should not happen if previous checks passed.")
                return pd.DataFrame()

            final_processed_rows = len(combined_df)
            st.success(f"Final processed DataFrame contains {final_processed_rows} rows for analysis.")
            
            if final_processed_rows == 0:
                st.error("After processing, no valid trade data remains. Please check your raw data and column names.")
            elif final_processed_rows < initial_combined_rows:
                st.info(f"Note: {initial_combined_rows - final_processed_rows} rows were dropped during processing due to missing or invalid data (e.g., date format, pre-2000 dates, non-numeric values in critical columns).")

            return combined_df
    st.warning("No files uploaded or no valid data could be processed.")
    return pd.DataFrame()

# --- Cached Function for ML Model Training ---
@st.cache_resource(show_spinner="Training AI model...")
def train_ml_model(df_trainable):
    """Trains a RandomForestClassifier model and returns it with features and target."""
    if df_trainable.empty or 'Net_Profit' not in df_trainable.columns:
        if df_trainable.empty:
            st.error("Cannot train model: Data is empty after processing.")
        else:
            st.error("Cannot train model: 'Net_Profit' column not found after processing. This column is critical.")
        return None, None, None

    ml_df = df_trainable.copy()
    ml_df['Is_Win'] = (ml_df['Net_Profit'] > 0).astype(int)

    # Prepare features with robust checks
    features_to_consider = {
        'LOTS': 'Log_LOTS', # Source column: 'LOTS', derived feature: 'Log_LOTS'
        'Prev_Trade_Win': 'Prev_Trade_Win',
        'Rolling_Avg_Profit': 'Rolling_Avg_Profit',
        'DATE_Hour': 'Hour', # Source column: 'DATE', derived feature: 'Hour'
        'DATE_DayOfWeek': 'DayOfWeek' # Source column: 'DATE', derived feature: 'DayOfWeek'
    }

    # Initialize derived feature columns to 0 or appropriate defaults
    ml_df['Log_LOTS'] = 0
    ml_df['Prev_Trade_Win'] = 0
    ml_df['Rolling_Avg_Profit_feature'] = 0 # Use a temporary name to avoid conflict
    ml_df['Hour'] = 0
    ml_df['DayOfWeek'] = 0

    actual_features_for_model = []

    # Process LOTS
    if 'LOTS' in ml_df.columns and not ml_df['LOTS'].empty and ml_df['LOTS'].sum() > 0:
        ml_df['Log_LOTS'] = np.log1p(ml_df['LOTS'])
        actual_features_for_model.append('Log_LOTS')
    else:
        st.warning("Cannot use 'LOTS' as a feature for ML model: column missing, empty, or all zeros.")

    # Process Previous Trade Win
    if 'Is_Win' in ml_df.columns and not ml_df['Is_Win'].empty:
        ml_df['Prev_Trade_Win'] = ml_df['Is_Win'].shift(1).fillna(0)
        actual_features_for_model.append('Prev_Trade_Win')
    else:
        st.warning("Cannot use 'Prev_Trade_Win' as a feature: 'Is_Win' column missing or empty.")

    # Process Rolling Average Profit
    if 'Rolling_Avg_Profit' in ml_df.columns and not ml_df['Rolling_Avg_Profit'].empty:
        ml_df['Rolling_Avg_Profit_feature'] = ml_df['Rolling_Avg_Profit'].shift(1).fillna(0)
        actual_features_for_model.append('Rolling_Avg_Profit_feature')
    elif 'Net_Profit' in ml_df.columns and not ml_df['Net_Profit'].empty:
        # Fallback: calculate rolling avg profit if not already done
        ml_df['Rolling_Avg_Profit_feature'] = ml_df['Net_Profit'].rolling(window=5, min_periods=1).mean().shift(1).fillna(0)
        actual_features_for_model.append('Rolling_Avg_Profit_feature')
        st.info("Calculated 'Rolling_Avg_Profit_feature' on the fly for ML model.")
    else:
        st.warning("Cannot use 'Rolling_Avg_Profit' as a feature: 'Rolling_Avg_Profit' or 'Net_Profit' column missing or empty.")

    # Process DATE derived features
    if 'DATE' in ml_df.columns and not ml_df['DATE'].empty:
        try:
            ml_df['Hour'] = ml_df['DATE'].dt.hour
            actual_features_for_model.append('Hour')
            ml_df['DayOfWeek'] = ml_df['DATE'].dt.dayofweek
            actual_features_for_model.append('DayOfWeek')
        except Exception as e:
            st.warning(f"Could not derive Hour/DayOfWeek from 'DATE' column: {e}. Skipping these features.")
    else:
        st.warning("Cannot use 'Hour' or 'DayOfWeek' as features: 'DATE' column missing or invalid.")
    
    # Remove duplicates from actual_features_for_model if any
    actual_features_for_model = list(dict.fromkeys(actual_features_for_model))

    if not actual_features_for_model:
        st.error("No suitable features could be derived from your data to train the ML model. Please ensure columns like 'DATE', 'LOTS', 'Net_Profit' are present and correctly processed.")
        return None, None, None
    
    # Ensure all feature columns are numeric, filling any remaining NaNs with 0
    X = ml_df[actual_features_for_model].apply(pd.to_numeric, errors='coerce').fillna(0)
    y = ml_df['Is_Win']

    # Drop any remaining NaNs in features or target before splitting
    combined_xy = pd.concat([X, y], axis=1).dropna()
    X_clean = combined_xy[actual_features_for_model]
    y_clean = combined_xy['Is_Win']

    if X_clean.empty or len(X_clean) < 2:
        st.warning(f"Not enough clean data ({len(X_clean)} rows) to train the prediction model after cleaning. Model training skipped.")
        return None, None, None
    
    try:
        # Check if both classes are present in y_clean
        if len(y_clean.unique()) < 2:
            st.warning("Only one class (all wins or all losses) found in data. Cannot train classification model. Need both wins and losses. Model training skipped.")
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
        feature_importances = pd.Series(model.feature_importances_, index=actual_features_for_model).sort_values(ascending=False)
        fig_feat_imp = px.bar(feature_importances, 
                              x=feature_importances.index, y=feature_importances.values,
                              title="Impact of Features on Trade Win/Loss Prediction",
                              labels={'x': 'Feature', 'y': 'Importance'},
                              color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_feat_imp.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig_feat_imp, use_container_width=True)

        return model, actual_features_for_model, y_clean.name
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.info("Check if your data has enough variety (both wins and losses) and if essential columns are present and correctly formatted.")
        return None, None, None


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
        "6. Simulated Trade Execution & Self-Improvement"
    ]
)
st.sidebar.markdown("---")
st.sidebar.info("This is a prototype AI Trading Assistant. Always perform your own due diligence.")

# --- Add a cache clearing button to the sidebar for easy access ---
if st.sidebar.button("Clear App Cache & Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.df = pd.DataFrame() # Clear the main DataFrame
    st.session_state.ml_model = None # Clear trained model
    st.session_state.simulated_trade_log = [] # Clear simulated log
    st.rerun() # Rerun the app to reflect changes
    st.success("Cache and all session data cleared! Please re-upload your data.")


# --- Page Content based on Navigation ---

# --- Welcome & Overview ---
if page_selection == "Welcome & Overview":
    st.markdown('<div class="welcome-section">', unsafe_allow_html=True)
    st.markdown("<h1>AI Trading Assistant</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p>Empowering your trading decisions with data-driven insights and artificial intelligence.</p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.subheader("What This App Offers:")
    st.markdown("""
    This application is designed to help you analyze your trading performance, identify key patterns, and receive AI-driven insights.
    
    **How to use:**
    1.  **Upload Your Trade Data:** Begin by uploading your historical trade data in CSV or XLSX format. The app is smart enough to process multiple files and all sheets within an Excel workbook!
    2.  **Explore Performance:** Get a comprehensive visualization of your equity curve, profit distribution, and rolling performance metrics.
    3.  **Train AI Model:** An Artificial Intelligence model will be trained on your historical data to predict the outcome (win/loss) of future trades.
    4.  **Get AI Signals:** Input current trade conditions and an optional chart image to receive a hypothetical AI-generated trade signal.
    5.  **Review Tips:** Access personalized improvement tips and corrective actions derived from your trading data and the AI model's analysis.
    6.  **Simulated Trade & Self-Improvement:** Understand how an AI agent would execute trades and explore the conceptual framework for AI self-learning.
    """)
    st.info("Navigate through the sections using the sidebar on the left to unlock full capabilities.")

# ---- SECTION 1: Upload Trade Data & Performance Analysis ----
elif page_selection == "1. Data Upload & Performance Analysis":
    st.header("1. Upload Your Trade Data & Performance Analysis")
    st.markdown("Upload your trade history files (CSV or XLSX). You can upload multiple files, and if an Excel file has multiple sheets, **all sheets will be analyzed**.")
    
    with st.container(border=True):
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
                
                if 'Net_Profit' in df.columns and not df['Net_Profit'].empty:
                    total_net_profit = df['Net_Profit'].sum()
                    total_trades = len(df)
                    
                    winning_trades = df[df['Net_Profit'] > 0]
                    losing_trades = df[df['Net_Profit'] <= 0]

                    win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
                    
                    avg_win = winning_trades['Net_Profit'].mean() if not winning_trades.empty else 0
                    avg_loss = losing_trades['Net_Profit'].mean() if not losing_trades.empty else 0

                    total_gross_profit = winning_trades['Net_Profit'].sum()
                    total_gross_loss = abs(losing_trades['Net_Profit'].sum())
                    
                    profit_factor = 0.0
                    try:
                        if total_gross_loss > 0:
                            profit_factor = total_gross_profit / total_gross_loss
                        elif total_gross_profit > 0 and total_gross_loss == 0:
                            profit_factor = np.inf # Infinite if no losses but has profit
                    except ZeroDivisionError:
                        profit_factor = np.inf if total_gross_profit > 0 else 0.0 # Handle case where both are zero

                    with col1:
                        st.metric(label="Total Net P/L", value=f"${total_net_profit:,.2f}")
                        st.metric(label="Win Rate", value=f"{win_rate:,.2f}%")
                    with col2:
                        st.metric(label="Total Trades", value=f"{total_trades}")
                        st.metric(label="Profit Factor", value=f"{profit_factor:,.2f}" if profit_factor != np.inf else "Infinity")
                    with col3:
                        st.metric(label="Avg Winning Trade", value=f"${avg_win:,.2f}")
                        st.metric(label="Avg Losing Trade", value=f"${avg_loss:,.2f}")
                    
                    if 'DATE' in df.columns and not df.empty and 'Cumulative_Profit' in df.columns:
                        max_drawdown = df['Drawdown'].min() if not df['Drawdown'].empty else 0
                        st.metric(label="Maximum Drawdown", value=f"${max_drawdown:,.2f}")
                    else:
                        st.warning("⚠️ 'DATE' and 'Net_Profit' columns are needed to calculate Max Drawdown.")
                else:
                    st.warning("Cannot calculate performance metrics. 'Net_Profit' column is missing or empty after processing.")

            st.markdown("---")
            with st.expander("View Performance Charts", expanded=True):
                if 'DATE' in df.columns and 'Cumulative_Profit' in df.columns and not df.empty:
                    st.subheader("Equity Curve")
                    fig_equity = px.line(df, x='DATE', y='Cumulative_Profit', 
                                         title="Cumulative Profit Over Time",
                                         labels={'DATE': 'Date', 'Cumulative_Profit': 'Cumulative Profit ($)'})
                    st.plotly_chart(fig_equity, use_container_width=True)
                elif 'DATE' not in df.columns:
                    st.info("Equity Curve cannot be generated: 'DATE' column is missing or invalid.")
                elif 'Cumulative_Profit' not in df.columns:
                     st.info("Equity Curve cannot be generated: 'Net_Profit' data is missing or invalid.")
                else:
                    st.info("No data available to generate Equity Curve.")

                if 'Net_Profit' in df.columns and not df['Net_Profit'].empty:
                    st.subheader("Profit Distribution")
                    fig_profit = px.histogram(df, x='Net_Profit', nbins=30, 
                                            title="Net Profit/Loss Distribution",
                                            labels={'Net_Profit': 'Net Profit/Loss'},
                                            color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig_profit, use_container_width=True)
                else:
                    st.info("Profit Distribution cannot be generated: 'Net_Profit' data is missing or empty.")


                if 'DATE' in df.columns and 'Rolling_Avg_Profit' in df.columns and 'Rolling_Avg_Win_Rate' in df.columns and not df.empty:
                    st.subheader("📊 Rolling Performance Over Time")
                    
                    fig_rolling_profit = px.line(df, x='DATE', y='Rolling_Avg_Profit',
                                                 title="Rolling Average Profit (Last 10 Trades)",
                                                 labels={'DATE': 'Date', 'Rolling_Avg_Profit': 'Avg Profit ($)'})
                    st.plotly_chart(fig_rolling_profit, use_container_width=True)

                    fig_rolling_win_rate = px.line(df, x='DATE', y='Rolling_Avg_Win_Rate',
                                                   title="Rolling Win Rate (Last 10 Trades)",
                                                   labels={'DATE': 'Date', 'Rolling_Avg_Win_Rate': 'Win Rate (%)'})
                    st.plotly_chart(fig_rolling_win_rate, use_container_width=True)

                    current_rolling_profit = df['Rolling_Avg_Profit'].iloc[-1] if not df['Rolling_Avg_Profit'].empty else 0
                    current_rolling_win_rate = df['Rolling_Avg_Win_Rate'].iloc[-1] if not df['Rolling_Avg_Win_Rate'].empty else 0
                    overall_avg_profit = df['Net_Profit'].mean() if 'Net_Profit' in df.columns and not df['Net_Profit'].empty else 0
                    overall_win_rate = (df['Net_Profit'] > 0).mean() * 100 if 'Net_Profit' in df.columns and not df['Net_Profit'].empty else 0

                    st.markdown("---")
                    st.subheader("Insights from Rolling Performance:")
                    if current_rolling_profit > overall_avg_profit * 1.1 and current_rolling_win_rate > overall_win_rate * 1.1:
                        st.success("🔥 **Hot Streak!** Your recent performance is significantly better than your overall average. Maintain discipline but consider if conditions are exceptionally favorable.")
                    elif current_rolling_profit < overall_avg_profit * 0.9 and current_rolling_win_rate < overall_win_rate * 0.9:
                        st.warning("❄️ **Cold Streak?** Your recent performance is below your overall average. It might be a good time to review your strategy, reduce position sizes, or consider taking a short break.")
                    else:
                        st.info("⚖️ Your recent performance is consistent with your overall average. Continue monitoring and adhering to your strategy.")
                    st.markdown("---")
                else:
                    st.info("Rolling Performance charts cannot be generated: Missing 'DATE', 'Net_Profit', or derived rolling columns, or no data.")


                if 'PAIR' in df.columns and not df['PAIR'].empty and 'Net_Profit' in df.columns and not df['Net_Profit'].empty:
                    st.subheader("Profitability by PAIR")
                    pair_profit = df.groupby('PAIR')['Net_Profit'].sum().sort_values(ascending=False)
                    fig_pair_profit = px.bar(pair_profit, x=pair_profit.index, y=pair_profit.values,
                                            title="Total Net Profit by PAIR",
                                            labels={'x': 'PAIR', 'y': 'Total Net Profit ($)'},
                                            color=pair_profit.values,
                                            color_continuous_scale=px.colors.sequential.Viridis)
                    st.plotly_chart(fig_pair_profit, use_container_width=True)
                else:
                    st.info("Profitability by PAIR cannot be generated: 'PAIR' or 'Net_Profit' column missing or empty.")

                if 'ORDER' in df.columns and not df['ORDER'].empty and 'Net_Profit' in df.columns and not df['Net_Profit'].empty:
                    st.subheader("Performance by Order Type (BUY/SELL)")
                    order_performance = df.groupby('ORDER')['Net_Profit'].agg(['sum', 'mean', lambda x: (x > 0).mean() * 100])
                    order_performance.columns = ['Total Profit', 'Avg Profit', 'Win Rate %']
                    st.dataframe(order_performance.round(2))
                else:
                    st.info("Performance by Order Type cannot be generated: 'ORDER' or 'Net_Profit' column missing or empty.")

                if 'DATE' in df.columns and not df['DATE'].empty and 'Net_Profit' in df.columns and not df['Net_Profit'].empty:
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
                        st.warning("Could not calculate best hour from 'DATE' and 'Net_Profit' data (results empty).")
                else:
                    st.session_state.best_hour = "N/A" 
                    st.info("💡 Tip: Ensure your 'DATE' and 'Net_Profit' columns are correctly formatted for time-based strategy suggestions.")
            
        else: # df is empty after processing
            st.warning("Please upload your trade data, or check the console for errors during data processing if no data is displayed here.")

    else: # If no files are uploaded initially, reset df
        st.session_state.df = pd.DataFrame()


# ---- SECTION 2: AI Model Training ----
elif page_selection == "2. AI Model Training":
    st.header("2. AI Model Training")
    st.info("The model learns from your historical data to predict if a trade will be a win or loss.")
    
    if not st.session_state.df.empty:
        if 'Net_Profit' in st.session_state.df.columns and len(st.session_state.df['Net_Profit'].unique()) > 1 and len(st.session_state.df) >= 10: # Ensure enough data and both classes
            if st.button("Train Prediction Model", key="train_model_button"):
                st.session_state.ml_model, st.session_state.ml_features, st.session_state.ml_target = train_ml_model(st.session_state.df)
        elif len(st.session_state.df) < 10:
             st.warning(f"Model training requires at least 10 trades in your data (you have {len(st.session_state.df)}). Please upload more data.")
        else:
            st.warning("Model training requires the 'Net_Profit' column to have both positive and negative values (wins and losses). Please ensure your data meets this requirement.")
        
        # Display model info if already trained
        if st.session_state.ml_model:
            st.subheader("Current Model Status:")
            st.success("Model is trained and ready for predictions!")
            if st.session_state.ml_features:
                st.write(f"**Features used:** {', '.join(st.session_state.ml_features)}")
    else:
        st.info("Please upload your trade data in the '1. Data Upload & Performance Analysis' section first to train the model.")


# ---- SECTION 3: Upload Chart Image ----
elif page_selection == "3. Chart Visual Analysis":
    st.header("3. Chart Visual Analysis")
    st.markdown("Upload a chart image to get a basic 'edge density' analysis, which is a placeholder for more advanced chart pattern recognition.")
    
    with st.container(border=True):
        uploaded_img = st.file_uploader("Upload Chart Image", type=["jpg", "jpeg", "png"], key="img_uploader")

    if uploaded_img:
        try:
            image = Image.open(uploaded_img).convert("RGB")
            st.image(image, caption="Uploaded Chart", use_column_width=True)
            image_np = np.array(image)

            st.subheader("Chart Analysis (Simplified AI)")
            # Check if image_np is not empty before processing
            if image_np.size > 0 and image_np.shape[0] > 1 and image_np.shape[1] > 1: # Ensure image has dimensions
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                
                edges = cv2.Canny(gray, 50, 150)
                st.image(edges, caption="Detected Edges (Potential Trendlines/Patterns)", use_column_width=True)

                total_pixels = edges.shape[0] * edges.shape[1]
                edge_pixels = np.sum(edges > 0)
                st.session_state.chart_feature_value = edge_pixels / total_pixels 
                st.info(f"**Calculated Chart Edge Density (as a simple feature):** {round(st.session_state.chart_feature_value * 100, 2)}%")
                st.warning("⚠️ **Note:** For true chart pattern recognition (e.g., Head & Shoulders, Flags), you would need to train a specialized Convolutional Neural Network (CNN) model. This 'edge density' is a very basic placeholder feature.")
            else:
                st.warning("Uploaded image is empty or corrupted. Cannot perform analysis.")
                st.session_state.chart_feature_value = 0.0
        except Exception as e:
            st.error(f"Error processing image: {e}. Please ensure it's a valid image file.")
            st.session_state.chart_feature_value = 0.0 
    else:
        st.info("Upload a chart image to analyze its edge density.")

# ---- SECTION 4: AI Agent Decision & Signal Generation ----
elif page_selection == "4. AI Trade Signal":
    st.header("4. AI Agent's Trade Signal & Prediction")
    st.markdown("Receive a hypothetical trade signal based on your trained AI model and current market conditions.")

    if st.session_state.ml_model and st.session_state.ml_features and st.session_state.df is not None and not st.session_state.df.empty:
        st.subheader("Provide Current Trade Conditions:")

        with st.container(border=True):
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
                    # Default based on last trade in historical data, if available
                    last_trade_win = st.session_state.df['Net_Profit'].iloc[-1] > 0 if 'Net_Profit' in st.session_state.df.columns and not st.session_state.df['Net_Profit'].empty else 0
                    input_data['Prev_Trade_Win'] = st.selectbox("Was the Previous Trade a Win?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=int(last_trade_win), key="pred_prev_win_input")
                elif feature == 'Rolling_Avg_Profit_feature': # Note the feature name change
                    # Use actual last rolling avg if available, otherwise default input
                    last_rolling_avg = st.session_state.df['Rolling_Avg_Profit'].iloc[-1] if 'Rolling_Avg_Profit' in st.session_state.df.columns and not st.session_state.df['Rolling_Avg_Profit'].empty else 0.0
                    input_data['Rolling_Avg_Profit_feature'] = st.number_input("Average Profit of Last 10 Trades", value=float(last_rolling_avg), step=1.0, key="pred_rolling_avg_input")
                # Add more robust input for other potential features if they are added later

        predict_df = pd.DataFrame([input_data])

        # Ensure all model features are present in predict_df, fill missing with 0
        for feature in st.session_state.ml_features:
            if feature not in predict_df.columns:
                predict_df[feature] = 0

        # Reorder columns to match the trained model's feature order
        predict_df = predict_df[st.session_state.ml_features]

        try:
            # Ensure predict_df contains only numeric values
            predict_df = predict_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            if not predict_df.empty:
                win_proba = st.session_state.ml_model.predict_proba(predict_df)[:, 1][0]
                prediction = st.session_state.ml_model.predict(predict_df)[0]

                st.markdown(f"**Probability of Winning Next Trade:** <span style='font-size: 1.2rem; font-weight: bold; color: {'#28a745' if win_proba > 0.65 else ('#ffc107' if win_proba > 0.35 else '#dc3545')}'>{round(win_proba * 100, 2)}%</span>", unsafe_allow_html=True)


                if prediction == 1 and win_proba > 0.65:
                    st.session_state.current_trade_signal = "🟢 **Consider BUY/LONG** (High Win Probability)"
                elif prediction == 0 and win_proba < 0.35:
                    st.session_state.current_trade_signal = "🔴 **Consider SELL/SHORT or AVOID** (High Loss Probability)"
                else:
                    st.session_state.current_trade_signal = "⚪ **HOLD / Neutral** (Uncertain or Moderate Probability)"

                st.markdown(f"### AI Trade Signal: {st.session_state.current_trade_signal}")

                with st.expander("Factors Contributing to This Signal"):
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
            else:
                st.error("Prediction input DataFrame is empty after cleaning. Cannot generate signal.")
                st.session_state.current_trade_signal = "Error: Input data for signal generation is empty."

        except Exception as e:
            st.error(f"Could not generate prediction. Ensure all required features for the model are available and valid input. Error: {e}")
            st.info(f"Model expected features: {st.session_state.ml_features}")
            st.session_state.current_trade_signal = "Error: Could not generate signal." 
    else:
        st.info("Please upload CSV/XLSX data in '1. Data Upload', train the model in '2. AI Model Training', and ensure your processed data is not empty to enable AI predictions.")

# ---- SECTION 5: Corrective Actions & Improvement Tips ----
elif page_selection == "5. Improvement Tips & Actions":
    st.header("5. Corrective Actions & Improvement Tips")
    st.markdown("Receive personalized insights and actionable tips based on your trading performance and the AI model's analysis.")

    if not st.session_state.df.empty and 'Net_Profit' in st.session_state.df.columns and not st.session_state.df['Net_Profit'].empty:
        df = st.session_state.df # Use a local variable for convenience
        
        with st.container(border=True):
            st.subheader("Data-Driven Insights:")

            if 'DATE' in df.columns and not df['DATE'].empty and 'Net_Profit' in df.columns and not df['Net_Profit'].empty:
                df['DayOfWeek'] = df['DATE'].dt.dayofweek
                day_of_week_profit = df.groupby('DayOfWeek')['Net_Profit'].sum()
                if not day_of_week_profit.empty:
                    best_day_of_week_num = day_of_week_profit.idxmax()
                    worst_day_of_week_num = day_of_week_profit.idxmin()
                    days_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                    st.info(f"💡 **Trading Day Performance:** You tend to perform best on **{days_map.get(best_day_of_week_num, 'N/A')}** and worst on **{days_map.get(worst_day_of_week_num, 'N/A')}**.")
                    if day_of_week_profit[worst_day_of_week_num] < 0:
                        st.warning(f"⚠️ **Corrective Action:** Consider reducing trading activity or reviewing your strategy specifically on {days_map.get(worst_day_of_week_num, 'N/A')}s, as it's been a net loss day.")
                else:
                    st.info("Could not determine best/worst trading day: Grouped data for DayOfWeek is empty.")
                
                st.info("📊 **Tip for Trade Duration:** If you track exact entry and exit times, we could analyze if shorter or longer trades are more profitable for you, leading to optimization of holding periods.")
            else:
                st.info("Daily performance insights cannot be generated: 'DATE' or 'Net_Profit' column missing or empty.")


            # Analysis based on ML model insights
            if st.session_state.ml_model and st.session_state.ml_features:
                st.subheader("Insights from Prediction Model:")
                feature_importances = pd.Series(st.session_state.ml_model.feature_importances_, index=st.session_state.ml_features).sort_values(ascending=False)
                if not feature_importances.empty:
                    top_feature = feature_importances.index[0]
                    st.write(f"- Your prediction model suggests that **'{top_feature}'** is the most influential factor in determining if your trades will be a win or loss.")
                    st.info(f"🔍 **Corrective Action:** Pay extra attention to '{top_feature}' when entering new trades. Try to optimize conditions around this factor.")
                else:
                    st.info("No feature importances to display (model might be empty or trained with no features).")
                
                if 'LOTS' in df.columns and 'Net_Profit' in df.columns and not df['LOTS'].empty and not df['Net_Profit'].empty: # Check if columns exist in the original df
                    try:
                        # Calculate correlation only if both columns are numeric and non-zero variance
                        if df['LOTS'].std() > 0 and df['Net_Profit'].std() > 0:
                            correlation = df['Net_Profit'].corr(df['LOTS'])
                            if correlation is not None: # Check for cases where correlation might be NaN if data is constant
                                if correlation < 0:
                                    st.warning(f"⚠️ **Corrective Action:** It appears larger 'LOTS' sizes ({correlation:.2f} correlation) might be negatively correlated with your profitability. Consider reviewing your position sizing strategy.")
                                elif correlation > 0.1: # Small positive correlation is fine
                                    st.info(f"✅ Your position sizing ('LOTS') seems to be generally aligned with profitability trends ({correlation:.2f} correlation).")
                                else:
                                    st.info(f"💡 'LOTS' vs 'Profit' correlation is low ({correlation:.2f}). This factor might not strongly influence your profitability.")
                            else:
                                st.info("💡 Could not calculate 'LOTS' vs 'Profit' correlation (result was NaN).")
                        else:
                             st.info("💡 Could not analyze 'LOTS' vs 'Profit' correlation due to insufficient variance in data (LOTS or Net_Profit values are too similar).")
                    except Exception as e:
                        st.info(f"Could not analyze LOTS vs Profit correlation: {e}")
                else:
                    st.info("LOTS vs Profitability correlation cannot be analyzed: 'LOTS' or 'Net_Profit' column missing or empty.")

            else:
                st.info("Train the AI model in '2. AI Model Training' to get model-driven tips.")
    else:
        st.info("Upload and process your trade data in '1. Data Upload' to generate detailed analysis and corrective actions.")

# ---- SECTION 6: Simulated Trade Execution (Conceptual) ----
elif page_selection == "6. Simulated Trade Execution & Self-Improvement":
    st.header("6. Simulated Trade Execution & AI Self-Improvement")
    st.warning("🛑 **This section demonstrates AI agent actions and conceptual self-improvement. It does NOT execute real trades.**")

    if st.session_state.ml_model and st.session_state.current_trade_signal != "No data or signal generated yet." and "Error" not in st.session_state.current_trade_signal: 
        st.subheader("AI Agent's Simulated Actions:")
        st.write("Once a signal is generated, a real AI trading agent would typically take actions like placing orders, monitoring positions, and most importantly, **learning from outcomes.**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Simulate Trade Execution")
            
            if st.button("Simulate Action based on Signal", key="sim_action_button"):
                signal = st.session_state.current_trade_signal
                log_entry = {"time": time.strftime("%H:%M:%S"), "signal": signal}

                if "BUY/LONG" in signal: 
                    st.success("✅ **Simulated BUY Order Initiated!**")
                    log_entry["action"] = "Simulated BUY"
                elif "SELL/SHORT" in signal: 
                    st.error("❌ **Simulated SELL Order Initiated!**")
                    log_entry["action"] = "Simulated SELL"
                else:
                    st.info("⚪ **No Strong Action.** Signal suggests HOLD/Neutral. No trade simulated.")
                    log_entry["action"] = "Simulated HOLD"
                
                st.session_state.simulated_trade_log.append(log_entry)
                
                st.markdown(f"**Simulated Agent Action:** `{log_entry['action']}` at `{log_entry['time']}` based on current signal.")
                st.write("*(In a real scenario, this would interact with a broker API.)*")

        with col2:
            st.markdown("#### Simulated Trade Log")
            if st.session_state.simulated_trade_log:
                log_df = pd.DataFrame(st.session_state.simulated_trade_log)
                st.dataframe(log_df.tail(5), hide_index=True)
            else:
                st.info("No simulated trades logged yet. Click 'Simulate Action' to see entries.")
            
            if st.button("Clear Simulated Log", key="clear_sim_log_button"):
                st.session_state.simulated_trade_log = []
                st.info("Simulated trade log cleared.")
                st.rerun()


        st.markdown("---")
        st.subheader("How AI Self-Improvement Works (Conceptually):")
        st.markdown("""
        For an AI to truly "self-improve," it needs a continuous feedback loop and persistent memory:
        1.  **Prediction:** The AI makes a trade signal (as this app does).
        2.  **Execution (Simulated/Real):** The agent acts on the signal.
        3.  **Outcome Observation:** The actual result of the trade (win/loss, profit/loss) is recorded.
        4.  **Data Collection:** This new outcome data, along with the conditions under which the trade was made, is added to the historical dataset.
        5.  **Re-training:** Periodically, or when enough new data is accumulated, the AI model is re-trained on the *expanded and updated* dataset. This is how it "learns" from its successes and failures.
        6.  **Model Update:** The improved model replaces the old one, leading to better future predictions.

        **In this prototype:**
        * You manually perform "Data Collection" by uploading updated trade history files in '1. Data Upload'.
        * You manually trigger "Re-training" in '2. AI Model Training'.
        * The **"AI Improves"** by learning from the new patterns in your updated data when you re-train the model!
        
        For a fully autonomous, self-improving agent, you would integrate with a real-time data source and a database to store outcomes and periodically retrain models automatically.
        """)

    else:
        st.info("Please upload CSV/XLSX data, train the model, and generate a signal in the previous steps to enable simulated trade execution and discuss self-improvement features.")

st.markdown("---")
st.markdown("_This AI Trading Assistant is a foundational prototype. For robust automated trading, significant development in real-time data integration, advanced ML models, comprehensive risk management, and secure API connections is essential._")
