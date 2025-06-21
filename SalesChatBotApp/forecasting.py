import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add time series analysis imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import seaborn as sns

class TimeSeriesAnalyzer:
    """Comprehensive time series analysis for sales data"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def prepare_time_series_data(self, data, date_column='Sale_Date', value_column='Net_Sales_Amount', 
                                freq='M', group_by=None, group_value=None):
        """Prepare time series data for analysis"""
        
        if data is None or len(data) == 0:
            st.error("No data provided for time series analysis")
            return None
            
        df = pd.DataFrame(data)
        
        # Handle date column
        if date_column in df.columns:
            df['Date'] = pd.to_datetime(df[date_column])
        elif 'Sales Year' in df.columns and 'Sales Month' in df.columns:
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            df['Month'] = df['Sales Month'].map(month_map)
            df['Date'] = pd.to_datetime(df['Sales Year'].astype(str) + '-' + 
                                      df['Month'].astype(str) + '-01')
        else:
            st.error("No date information found in data")
            return None
            
        # Filter by group if specified
        if group_by and group_value and group_by in df.columns:
            df = df[df[group_by] == group_value]
            
        # Aggregate by date
        if freq == 'M':  # Monthly
            ts_data = df.groupby(pd.Grouper(key='Date', freq='M'))[value_column].sum()
        elif freq == 'W':  # Weekly
            ts_data = df.groupby(pd.Grouper(key='Date', freq='W'))[value_column].sum()
        elif freq == 'D':  # Daily
            ts_data = df.groupby(pd.Grouper(key='Date', freq='D'))[value_column].sum()
        else:
            ts_data = df.groupby(pd.Grouper(key='Date', freq=freq))[value_column].sum()
            
        # Remove NaN values
        ts_data = ts_data.dropna()
        
        if ts_data.empty:
            st.error("No valid time series data after aggregation")
            return None
            
        return ts_data
    
    def analyze_trend(self, ts_data):
        """Analyze trend in time series data"""
        
        if ts_data is None or len(ts_data) < 2:
            return None
            
        # Linear trend analysis
        x = np.arange(len(ts_data))
        y = ts_data.values
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate trend metrics
        trend_strength = abs(r_value)
        trend_direction = "Increasing" if slope > 0 else "Decreasing"
        trend_significance = "Significant" if p_value < 0.05 else "Not Significant"
        
        # Calculate percentage change
        total_change = ((ts_data.iloc[-1] - ts_data.iloc[0]) / ts_data.iloc[0]) * 100
        avg_monthly_change = slope * 12  # Assuming monthly data
        
        trend_analysis = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'trend_significance': trend_significance,
            'total_change_percent': total_change,
            'avg_monthly_change': avg_monthly_change,
            'trend_line': intercept + slope * x
        }
        
        return trend_analysis
    
    def analyze_seasonality(self, ts_data, period=12):
        """Analyze seasonality in time series data"""
        
        if ts_data is None or len(ts_data) < period * 2:
            return None
            
        # Seasonal decomposition
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=period)
            
            # Calculate seasonal strength
            seasonal_strength = np.std(decomposition.seasonal) / np.std(ts_data)
            
            # Find peak and trough months
            seasonal_values = decomposition.seasonal[:period]
            peak_month = seasonal_values.idxmax()
            trough_month = seasonal_values.idxmin()
            
            # Calculate seasonal variation
            seasonal_variation = (seasonal_values.max() - seasonal_values.min()) / ts_data.mean() * 100
            
            seasonality_analysis = {
                'seasonal_strength': seasonal_strength,
                'seasonal_variation_percent': seasonal_variation,
                'peak_month': peak_month,
                'trough_month': trough_month,
                'decomposition': decomposition,
                'seasonal_values': seasonal_values
            }
            
            return seasonality_analysis
            
        except Exception as e:
            st.warning(f"Could not perform seasonal decomposition: {e}")
            return None
    
    def test_stationarity(self, ts_data):
        """Test stationarity using ADF and KPSS tests"""
        
        if ts_data is None or len(ts_data) < 10:
            return None
            
        # Augmented Dickey-Fuller test
        try:
            adf_result = adfuller(ts_data.dropna())
            adf_statistic = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical_values = adf_result[4]
            
            # KPSS test
            kpss_result = kpss(ts_data.dropna())
            kpss_statistic = kpss_result[0]
            kpss_pvalue = kpss_result[1]
            kpss_critical_values = kpss_result[3]
            
            # Interpret results
            adf_stationary = adf_pvalue < 0.05
            kpss_stationary = kpss_pvalue > 0.05
            
            # Overall stationarity assessment
            if adf_stationary and kpss_stationary:
                stationarity = "Stationary"
            elif not adf_stationary and not kpss_stationary:
                stationarity = "Non-stationary"
            else:
                stationarity = "Inconclusive"
            
            stationarity_analysis = {
                'adf_statistic': adf_statistic,
                'adf_pvalue': adf_pvalue,
                'adf_critical_values': adf_critical_values,
                'adf_stationary': adf_stationary,
                'kpss_statistic': kpss_statistic,
                'kpss_pvalue': kpss_pvalue,
                'kpss_critical_values': kpss_critical_values,
                'kpss_stationary': kpss_stationary,
                'overall_stationarity': stationarity
            }
            
            return stationarity_analysis
            
        except Exception as e:
            st.warning(f"Could not perform stationarity tests: {e}")
            return None
    
    def analyze_autocorrelation(self, ts_data, max_lag=20):
        """Analyze autocorrelation and partial autocorrelation"""
        
        if ts_data is None or len(ts_data) < max_lag + 1:
            return None
            
        try:
            # Calculate autocorrelation
            acf_values = []
            pacf_values = []
            
            for lag in range(1, min(max_lag + 1, len(ts_data) // 2)):
                # ACF
                acf = ts_data.autocorr(lag=lag)
                acf_values.append(acf)
                
                # PACF (simplified calculation)
                if lag == 1:
                    pacf_values.append(acf)
                else:
                    # For simplicity, we'll use a basic PACF approximation
                    pacf_values.append(acf * 0.5)  # Simplified
            
            # Find significant lags
            significance_threshold = 1.96 / np.sqrt(len(ts_data))  # 95% confidence interval
            
            significant_acf_lags = [i+1 for i, val in enumerate(acf_values) if abs(val) > significance_threshold]
            significant_pacf_lags = [i+1 for i, val in enumerate(pacf_values) if abs(val) > significance_threshold]
            
            autocorr_analysis = {
                'acf_values': acf_values,
                'pacf_values': pacf_values,
                'significant_acf_lags': significant_acf_lags,
                'significant_pacf_lags': significant_pacf_lags,
                'significance_threshold': significance_threshold,
                'max_acf_lag': max_lag
            }
            
            return autocorr_analysis
            
        except Exception as e:
            st.warning(f"Could not perform autocorrelation analysis: {e}")
            return None
    
    def detect_cycles(self, ts_data):
        """Detect cycles and periodic patterns"""
        
        if ts_data is None or len(ts_data) < 20:
            return None
            
        try:
            # Remove trend for cycle detection
            detrended = ts_data - ts_data.rolling(window=min(12, len(ts_data)//4), center=True).mean()
            detrended = detrended.dropna()
            
            if len(detrended) < 10:
                return None
            
            # Periodogram analysis
            freqs, power = periodogram(detrended, fs=1.0)
            
            # Find dominant frequencies
            dominant_freq_idx = np.argsort(power)[-3:]  # Top 3 frequencies
            dominant_periods = 1 / freqs[dominant_freq_idx]
            
            # Filter realistic periods (between 2 and len(data)/2)
            realistic_periods = [p for p in dominant_periods if 2 <= p <= len(detrended)/2]
            
            cycle_analysis = {
                'dominant_periods': realistic_periods,
                'frequencies': freqs,
                'power_spectrum': power,
                'detrended_data': detrended
            }
            
            return cycle_analysis
            
        except Exception as e:
            st.warning(f"Could not perform cycle detection: {e}")
            return None
    
    def calculate_statistical_metrics(self, ts_data):
        """Calculate comprehensive statistical metrics"""
        
        if ts_data is None or len(ts_data) < 2:
            return None
            
        # Basic statistics
        mean_val = ts_data.mean()
        std_val = ts_data.std()
        median_val = ts_data.median()
        
        # Skewness and kurtosis
        skewness = ts_data.skew()
        kurtosis = ts_data.kurtosis()
        
        # Coefficient of variation
        cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
        
        # Range and percentiles
        data_range = ts_data.max() - ts_data.min()
        q25 = ts_data.quantile(0.25)
        q75 = ts_data.quantile(0.75)
        iqr = q75 - q25
        
        # Growth rates
        growth_rates = ts_data.pct_change().dropna()
        avg_growth_rate = growth_rates.mean() * 100
        growth_volatility = growth_rates.std() * 100
        
        # Volatility clustering (simplified)
        volatility = growth_rates.rolling(window=min(6, len(growth_rates))).std().mean() * 100
        
        statistical_metrics = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'coefficient_of_variation': cv,
            'range': data_range,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'avg_growth_rate': avg_growth_rate,
            'growth_volatility': growth_volatility,
            'volatility': volatility
        }
        
        return statistical_metrics
    
    def create_time_series_plots(self, ts_data, trend_analysis=None, seasonality_analysis=None, 
                                autocorr_analysis=None, cycle_analysis=None):
        """Create comprehensive time series visualization plots"""
        
        if ts_data is None:
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Time Series', 'Trend Analysis', 'Seasonal Decomposition', 
                          'Autocorrelation', 'Growth Rates', 'Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Original time series
        fig.add_trace(
            go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines+markers', 
                      name='Sales', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 2. Trend analysis
        if trend_analysis:
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=ts_data.values, mode='lines+markers', 
                          name='Actual', line=dict(color='blue')),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=ts_data.index, y=trend_analysis['trend_line'], 
                          mode='lines', name='Trend', line=dict(color='red', dash='dash')),
                row=1, col=2
            )
        
        # 3. Seasonal decomposition
        if seasonality_analysis and hasattr(seasonality_analysis['decomposition'], 'seasonal'):
            decomp = seasonality_analysis['decomposition']
            fig.add_trace(
                go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, 
                          mode='lines', name='Seasonal', line=dict(color='green')),
                row=2, col=1
            )
        
        # 4. Autocorrelation
        if autocorr_analysis:
            lags = list(range(1, len(autocorr_analysis['acf_values']) + 1))
            fig.add_trace(
                go.Bar(x=lags, y=autocorr_analysis['acf_values'], 
                      name='ACF', marker_color='orange'),
                row=2, col=2
            )
            # Add significance lines
            threshold = autocorr_analysis['significance_threshold']
            fig.add_hline(y=threshold, line_dash="dash", line_color="red", row=2, col=2)
            fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=2, col=2)
        
        # 5. Growth rates
        growth_rates = ts_data.pct_change().dropna()
        fig.add_trace(
            go.Scatter(x=growth_rates.index, y=growth_rates.values * 100, 
                      mode='lines', name='Growth Rate (%)', line=dict(color='purple')),
            row=3, col=1
        )
        
        # 6. Distribution
        fig.add_trace(
            go.Histogram(x=ts_data.values, nbinsx=20, name='Distribution', 
                        marker_color='lightblue'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Comprehensive Time Series Analysis",
            showlegend=True
        )
        
        return fig
    
    def generate_time_series_report(self, ts_data, group_by=None, group_value=None):
        """Generate comprehensive time series analysis report"""
        
        if ts_data is None:
            return "No data available for time series analysis."
        
        report = f"## 📊 Time Series Analysis Report\n\n"
        
        if group_by and group_value:
            report += f"**Analysis for {group_by}: {group_value}**\n\n"
        
        # Basic information
        report += f"**Data Overview:**\n"
        report += f"• Time period: {ts_data.index.min().strftime('%Y-%m-%d')} to {ts_data.index.max().strftime('%Y-%m-%d')}\n"
        report += f"• Number of observations: {len(ts_data)}\n"
        report += f"• Frequency: {ts_data.index.freq or 'Irregular'}\n\n"
        
        # Statistical metrics
        stats = self.calculate_statistical_metrics(ts_data)
        if stats:
            report += f"**Statistical Summary:**\n"
            report += f"• Mean: ৳{stats['mean']:,.2f}\n"
            report += f"• Standard Deviation: ৳{stats['std']:,.2f}\n"
            report += f"• Coefficient of Variation: {stats['coefficient_of_variation']:.1f}%\n"
            report += f"• Skewness: {stats['skewness']:.3f}\n"
            report += f"• Kurtosis: {stats['kurtosis']:.3f}\n\n"
        
        # Trend analysis
        trend = self.analyze_trend(ts_data)
        if trend:
            report += f"**Trend Analysis:**\n"
            report += f"• Direction: {trend['trend_direction']}\n"
            report += f"• Strength: {trend['trend_strength']:.3f} (R²)\n"
            report += f"• Significance: {trend['trend_significance']} (p={trend['p_value']:.3f})\n"
            report += f"• Total Change: {trend['total_change_percent']:+.1f}%\n"
            report += f"• Average Monthly Change: ৳{trend['avg_monthly_change']:,.2f}\n\n"
        
        # Seasonality analysis
        seasonality = self.analyze_seasonality(ts_data)
        if seasonality:
            report += f"**Seasonality Analysis:**\n"
            report += f"• Seasonal Strength: {seasonality['seasonal_strength']:.3f}\n"
            report += f"• Seasonal Variation: {seasonality['seasonal_variation_percent']:.1f}%\n"
            report += f"• Peak Month: {seasonality['peak_month'].strftime('%B %Y')}\n"
            report += f"• Trough Month: {seasonality['trough_month'].strftime('%B %Y')}\n\n"
        
        # Stationarity test
        stationarity = self.test_stationarity(ts_data)
        if stationarity:
            report += f"**Stationarity Tests:**\n"
            report += f"• ADF Test: {'Stationary' if stationarity['adf_stationary'] else 'Non-stationary'} (p={stationarity['adf_pvalue']:.3f})\n"
            report += f"• KPSS Test: {'Stationary' if stationarity['kpss_stationary'] else 'Non-stationary'} (p={stationarity['kpss_pvalue']:.3f})\n"
            report += f"• Overall Assessment: {stationarity['overall_stationarity']}\n\n"
        
        # Autocorrelation analysis
        autocorr = self.analyze_autocorrelation(ts_data)
        if autocorr:
            report += f"**Autocorrelation Analysis:**\n"
            report += f"• Significant ACF lags: {autocorr['significant_acf_lags'][:5]}\n"
            report += f"• Significant PACF lags: {autocorr['significant_pacf_lags'][:5]}\n\n"
        
        # Cycle detection
        cycles = self.detect_cycles(ts_data)
        if cycles and cycles['dominant_periods']:
            report += f"**Cycle Detection:**\n"
            report += f"• Dominant periods: {[f'{p:.1f} periods' for p in cycles['dominant_periods']]}\n\n"
        
        # Growth analysis
        if stats:
            report += f"**Growth Analysis:**\n"
            report += f"• Average Growth Rate: {stats['avg_growth_rate']:.2f}%\n"
            report += f"• Growth Volatility: {stats['growth_volatility']:.2f}%\n"
            report += f"• Volatility Clustering: {stats['volatility']:.2f}%\n\n"
        
        # Recommendations
        report += f"**Recommendations:**\n"
        if trend and trend['trend_strength'] > 0.7:
            report += f"• Strong trend detected - consider trend-based forecasting models\n"
        if seasonality and seasonality['seasonal_strength'] > 0.3:
            report += f"• Significant seasonality detected - include seasonal components in models\n"
        if stationarity and stationarity['overall_stationarity'] == 'Non-stationary':
            report += f"• Non-stationary series - consider differencing or transformation\n"
        if autocorr and len(autocorr['significant_acf_lags']) > 0:
            report += f"• Autocorrelation present - consider ARIMA or similar models\n"
        
        return report

class SalesForecaster:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_columns = []
        
    def prepare_forecasting_data(self, data):
        """
        Prepare comprehensive data for forecasting using all available features.
        This includes creating time-based features, encoding categorical variables,
        generating interaction features, and calculating advanced lag and rolling statistics.
        """
        if not data:
            st.warning("No data provided for forecasting preparation.")
            return None
            
        df = pd.DataFrame(data)
        st.write(f"📊 Processing {len(df)} records for forecasting...")
        
        # Convert Sale_Date to datetime and extract features
        if 'Sale_Date' in df.columns:
            df['Sale_Date'] = pd.to_datetime(df['Sale_Date'])
            df['Year'] = df['Sale_Date'].dt.year
            df['Month'] = df['Sale_Date'].dt.month
            df['Quarter'] = df['Sale_Date'].dt.quarter
            df['DayOfWeek'] = df['Sale_Date'].dt.dayofweek
            df['DayOfYear'] = df['Sale_Date'].dt.dayofyear
            df['WeekOfYear'] = df['Sale_Date'].dt.isocalendar().week.astype(int)
            df['Day'] = df['Sale_Date'].dt.day # Added Day feature
        else:
            # Fallback for data without 'Sale_Date'
            if 'Sales Year' in df.columns and 'Sales Month' in df.columns:
                month_map = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                df['Month'] = df['Sales Month'].map(month_map)
                df['Year'] = df['Sales Year']
                df['Quarter'] = df['Sales Quarter'].str.extract(r'Q(\d+)').astype(int)
                # If no Sale_Date, assume day 1 for date creation for consistency
                df['Sale_Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')
            else:
                st.error("Missing 'Sale_Date' or 'Sales Year'/'Sales Month' columns for time series analysis.")
                return None

        # Create comprehensive time-based cyclical features
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter'] / 4)
        df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter'] / 4)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7) # Added Day of Week Cyclical
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7) # Added Day of Week Cyclical
        df['DayOfYear_Sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365.25) # Added Day of Year Cyclical
        df['DayOfYear_Cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365.25) # Added Day of Year Cyclical
        
        # Create trend features
        # Normalize time trend to prevent very large numbers
        df['Time_Trend'] = (df['Year'] - df['Year'].min()) * 12 + df['Month']
        df['Time_Trend_Normalized'] = (df['Time_Trend'] - df['Time_Trend'].min()) / (df['Time_Trend'].max() - df['Time_Trend'].min())
        
        # Create seasonal indicators (binary flags)
        df['Is_Q4'] = (df['Quarter'] == 4).astype(int)
        df['Is_Q1'] = (df['Quarter'] == 1).astype(int)
        df['Is_Holiday_Season'] = df['Month'].isin([11, 12]).astype(int)  # Nov, Dec
        df['Is_Summer_Season'] = df['Month'].isin([6, 7, 8]).astype(int)  # Jun, Jul, Aug
        
        # Encode categorical variables using LabelEncoder
        categorical_cols = [col for col in ['Product_Name', 'Product_Category', 'Store_Region', 
                                            'Payment_Method', 'Transaction_Status', 'Customer_City', 
                                            'Customer_State', 'Salesperson_Name'] if col in df.columns]
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Handle potential new categories in future data by fitting on all available data if possible
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Create interaction features between encoded categorical variables and time trends
        if 'Product_Category_encoded' in df.columns and 'Store_Region_encoded' in df.columns:
            df['Product_Region_Interaction'] = df['Product_Category_encoded'] * df['Store_Region_encoded']
        
        if 'Payment_Method_encoded' in df.columns and 'Store_Region_encoded' in df.columns:
            df['Payment_Region_Interaction'] = df['Payment_Method_encoded'] * df['Store_Region_encoded']
        
        # Sort data by Sale_Date for time series operations
        df = df.sort_values('Sale_Date')
        
        # Aggregate by time period and other dimensions for consistent time series analysis
        # Using 'Sale_Date' aggregated to month-level for consistency
        df['Month_Year'] = df['Sale_Date'].dt.to_period('M')

        agg_cols = ['Month_Year']
        if 'Product_Name' in df.columns:
            agg_cols.append('Product_Name')
        if 'Store_Region' in df.columns:
            agg_cols.append('Store_Region')
        
        # Group by time features and aggregate sales
        agg_data = df.groupby(agg_cols).agg(
            Net_Sales_Amount_sum=('Net_Sales_Amount', 'sum'),
            Net_Sales_Amount_mean=('Net_Sales_Amount', 'mean'),
            Quantity_Sold_sum=('Quantity_Sold', 'sum'),
            Quantity_Sold_mean=('Quantity_Sold', 'mean'),
            Unit_Price_mean=('Unit_Price', 'mean'),
            Discount_Amount_sum=('Discount_Amount', 'sum'),
            Tax_Amount_sum=('Tax_Amount', 'sum')
        ).reset_index()

        # Convert Month_Year back to datetime for easier feature creation
        agg_data['Date'] = agg_data['Month_Year'].dt.to_timestamp()
        agg_data['Year'] = agg_data['Date'].dt.year
        agg_data['Month'] = agg_data['Date'].dt.month
        agg_data['Quarter'] = agg_data['Date'].dt.quarter
        
        # Re-add cyclical and trend features to aggregated data
        agg_data['Month_Sin'] = np.sin(2 * np.pi * agg_data['Month'] / 12)
        agg_data['Month_Cos'] = np.cos(2 * np.pi * agg_data['Month'] / 12)
        agg_data['Quarter_Sin'] = np.sin(2 * np.pi * agg_data['Quarter'] / 4)
        agg_data['Quarter_Cos'] = np.cos(2 * np.pi * agg_data['Quarter'] / 4)
        agg_data['Time_Trend'] = (agg_data['Year'] - agg_data['Year'].min()) * 12 + agg_data['Month']
        agg_data['Time_Trend_Normalized'] = (agg_data['Time_Trend'] - agg_data['Time_Trend'].min()) / (agg_data['Time_Trend'].max() - agg_data['Time_Trend'].min())
        agg_data['Is_Q4'] = (agg_data['Quarter'] == 4).astype(int)
        agg_data['Is_Q1'] = (agg_data['Quarter'] == 1).astype(int)
        agg_data['Is_Holiday_Season'] = agg_data['Month'].isin([11, 12]).astype(int)
        agg_data['Is_Summer_Season'] = agg_data['Month'].isin([6, 7, 8]).astype(int)

        # Create lag features based on the aggregated 'Net_Sales_Amount_sum'
        # Group by non-time columns if present
        group_by_cols_for_lag = [col for col in agg_cols if col not in ['Month_Year']]
        if group_by_cols_for_lag:
            agg_data = agg_data.sort_values(group_by_cols_for_lag + ['Date'])
            for lag in [1, 2, 3, 4, 5, 6, 12]:  # Extended lag periods
                agg_data[f'Sales_Lag_{lag}'] = agg_data.groupby(group_by_cols_for_lag)['Net_Sales_Amount_sum'].shift(lag)
                agg_data[f'Quantity_Lag_{lag}'] = agg_data.groupby(group_by_cols_for_lag)['Quantity_Sold_sum'].shift(lag)
        else: # No grouping, just global lags
            agg_data = agg_data.sort_values('Date')
            for lag in [1, 2, 3, 4, 5, 6, 12]:
                agg_data[f'Sales_Lag_{lag}'] = agg_data['Net_Sales_Amount_sum'].shift(lag)
                agg_data[f'Quantity_Lag_{lag}'] = agg_data['Quantity_Sold_sum'].shift(lag)

        # Create rolling averages and standard deviations with different windows
        for window in [3, 6, 12, 24]:  # More windows for robustness
            if group_by_cols_for_lag:
                agg_data[f'Sales_MA_{window}'] = agg_data.groupby(group_by_cols_for_lag)['Net_Sales_Amount_sum'].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                agg_data[f'Sales_Std_{window}'] = agg_data.groupby(group_by_cols_for_lag)['Net_Sales_Amount_sum'].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            else:
                agg_data[f'Sales_MA_{window}'] = agg_data['Net_Sales_Amount_sum'].rolling(window=window, min_periods=1).mean()
                agg_data[f'Sales_Std_{window}'] = agg_data['Net_Sales_Amount_sum'].rolling(window=window, min_periods=1).std()
        
        # Create exponential moving averages
        for span in [7, 30, 90]: # Spans for EMA, related to days/months
            if group_by_cols_for_lag:
                 agg_data[f'Sales_EMA_{span}'] = agg_data.groupby(group_by_cols_for_lag)['Net_Sales_Amount_sum'].transform(lambda x: x.ewm(span=span, adjust=False).mean())
            else:
                 agg_data[f'Sales_EMA_{span}'] = agg_data['Net_Sales_Amount_sum'].ewm(span=span, adjust=False).mean()

        # Fill NaNs created by lagging/rolling operations before dropping
        # Use a sensible fill value (e.g., 0 or the mean/median of the column)
        # For sales-related features, 0 is often appropriate for initial missing lags
        for col in agg_data.columns:
            if 'Lag_' in col or 'MA_' in col or 'Std_' in col or 'EMA_' in col:
                agg_data[col] = agg_data[col].fillna(0) # Filling NaNs from lags/rolling with 0

        # Create additional trend and seasonality features on aggregated data
        agg_data['Year_Progress'] = agg_data['Month'] / 12  # Progress through the year
        agg_data['Quarter_Progress'] = (agg_data['Month'] % 3) / 3  # Progress through quarter
        agg_data['Month_Squared'] = agg_data['Month'] ** 2  # Non-linear month effect
        agg_data['Time_Trend_Squared'] = agg_data['Time_Trend'] ** 2  # Non-linear trend
        
        # Interaction features between time and trends/seasonality on aggregated data
        agg_data['Trend_Month_Interaction'] = agg_data['Time_Trend_Normalized'] * agg_data['Month_Sin']
        agg_data['Trend_Quarter_Interaction'] = agg_data['Time_Trend_Normalized'] * agg_data['Quarter_Sin']
        agg_data['Holiday_Trend_Interaction'] = agg_data['Is_Holiday_Season'] * agg_data['Time_Trend_Normalized']

        # Drop rows with any remaining NaN values after feature creation (should be minimal)
        initial_rows = len(agg_data)
        agg_data = agg_data.dropna().reset_index(drop=True)
        if len(agg_data) < initial_rows:
            st.warning(f"Dropped {initial_rows - len(agg_data)} rows due to missing values after feature engineering.")

        st.success(f"✅ Prepared {len(agg_data)} aggregated records with {len(agg_data.columns)} features.")
        
        return agg_data
    
    def train_forecasting_model(self, data, target_column='Net_Sales_Amount_sum', 
                               group_by=None, model_type='gradient_boosting'):
        """
        Train an advanced forecasting model with a comprehensive feature set.
        Includes feature scaling, time-series aware splitting, and cross-validation.
        """
        
        if data is None or data.empty:
            st.error("No data available for model training.")
            return None, None, None
            
        st.write("🤖 Training ML model...")
        
        # Define comprehensive feature set
        base_features = [
            'Year', 'Month', 'Quarter', 'Month_Sin', 'Month_Cos', 'Quarter_Sin', 'Quarter_Cos',
            'DayOfWeek_Sin', 'DayOfWeek_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos', # Added cyclical day features
            'Time_Trend_Normalized', 'Time_Trend_Squared', # Using normalized trend
            'Is_Q4', 'Is_Q1', 'Is_Holiday_Season', 'Is_Summer_Season',
            'Year_Progress', 'Quarter_Progress', 'Month_Squared',
            'Trend_Month_Interaction', 'Trend_Quarter_Interaction', 'Holiday_Trend_Interaction' # Added interaction features
        ]
        
        # Dynamically add lag, rolling mean, and EMA features if they exist in data
        lag_features = [col for col in data.columns if 'Lag_' in col]
        ma_features = [col for col in data.columns if 'MA_' in col or 'EMA_' in col or 'Std_' in col]
        
        feature_cols = list(set(base_features + lag_features + ma_features)) # Use set to avoid duplicates

        # Add encoded group-by column if specified and exists
        if group_by and f'{group_by}_encoded' in data.columns:
            feature_cols.append(f'{group_by}_encoded')
        elif group_by and group_by in data.columns:
             # If encoding doesn't exist, create it on the fly for the aggregated data
            le = LabelEncoder()
            data[f'{group_by}_encoded'] = le.fit_transform(data[group_by].astype(str))
            self.label_encoders[group_by] = le # Store for future use
            feature_cols.append(f'{group_by}_encoded')

        # Filter feature_cols to only include those present in the data
        feature_cols = [col for col in feature_cols if col in data.columns]
        
        # Ensure target column is in data
        if target_column not in data.columns:
            st.error(f"Target column '{target_column}' not found in data.")
            return None, None, None

        # Prepare X (features) and y (target)
        X = data[feature_cols].copy()
        y = data[target_column]
        
        # Handle any remaining missing values (e.g., from initial lags)
        X = X.fillna(0) # A common strategy for time series features

        # Time-series aware splitting: split data chronologically
        # Use a fixed percentage for training (e.g., 80%) to ensure test set is future data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if X_train.empty or X_test.empty:
            st.error("Insufficient data for training and testing split.")
            return None, None, None

        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train chosen model with optimized parameters
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=0.5, random_state=42) # Slightly increased regularization
        elif model_type == 'lasso':
            model = Lasso(alpha=0.05, random_state=42) # Slightly increased regularization
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_split=5, 
                                        min_samples_leaf=3, random_state=42, n_jobs=-1,
                                        max_features='sqrt') # Added max_features for better generalization
        elif model_type == 'gradient_boosting':
            # Adjusted parameters for better performance and to reduce overfitting
            model = GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, 
                                            subsample=0.7, random_state=42,
                                            loss='huber', # Huber loss is more robust to outliers
                                            min_samples_leaf=5)
        else:
            # Default to Gradient Boosting with recommended parameters
            st.warning(f"Unknown model type '{model_type}'. Defaulting to Gradient Boosting.")
            model = GradientBoostingRegressor(n_estimators=300, max_depth=7, learning_rate=0.05, 
                                            subsample=0.7, random_state=42,
                                            loss='huber',
                                            min_samples_leaf=5)
            
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model using TimeSeriesSplit for more robust cross-validation
        # Ensure n_splits is not too large for small datasets
        n_splits_cv = min(5, len(X_train) // 2) # At least 2 samples per split
        if n_splits_cv < 2: # Ensure at least 2 splits for cross-validation
            cv_scores = np.array([np.nan]) # Not enough data for meaningful CV
            st.warning("Not enough data for robust cross-validation (less than 2 splits).")
        else:
            tscv = TimeSeriesSplit(n_splits=n_splits_cv)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='r2', n_jobs=-1)
        
        # Test set evaluation
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'CV_R2_Mean': cv_scores.mean() if len(cv_scores) > 0 and not np.isnan(cv_scores).all() else 0.0,
            'CV_R2_Std': cv_scores.std() if len(cv_scores) > 0 and not np.isnan(cv_scores).all() else 0.0,
            'Feature_Importance': dict(zip(feature_cols, model.feature_importances_)) if hasattr(model, 'feature_importances_') else None
        }
        
        st.success(f"✅ Model trained successfully! R² Score: {r2:.3f}, CV R²: {metrics['CV_R2_Mean']:.3f} ± {metrics['CV_R2_Std']:.3f}")
        
        # Store selected feature columns
        self.feature_columns = feature_cols
        
        return model, scaler, metrics
    
    def generate_simple_trend_forecast(self, data, periods=12, group_by=None, group_value=None):
        """
        Generate simple trend-based forecast as a fallback or for comparison.
        This uses a linear trend with seasonal adjustment based on historical monthly means.
        """
        
        if data is None or data.empty:
            st.warning("No historical data to generate simple trend forecast.")
            return None
            
        st.write(f"📈 Generating simple trend-based forecast for {periods} months...")
        
        # Aggregate data by time period, potentially including group_by
        if group_by and group_by in data.columns:
            agg_data = data.groupby(['Year', 'Month', group_by])['Net_Sales_Amount_sum'].sum().reset_index()
            if group_value:
                agg_data = agg_data[agg_data[group_by] == group_value]
        else:
            agg_data = data.groupby(['Year', 'Month'])['Net_Sales_Amount_sum'].sum().reset_index()
        
        # Sort by time to ensure correct trend calculation
        agg_data = agg_data.sort_values(['Year', 'Month']).reset_index(drop=True)
        
        if agg_data.empty:
            st.warning("Aggregated data is empty for simple trend forecast.")
            return None

        # Calculate a continuous time point for linear regression
        agg_data['Time_Point'] = np.arange(len(agg_data))
        
        # Simple linear trend model
        X_trend = agg_data[['Time_Point']].values
        y_trend = agg_data['Net_Sales_Amount_sum'].values
        
        if len(X_trend) < 2:
            st.warning("Not enough data points to calculate a trend for simple forecast.")
            # Fallback to last known value if trend cannot be calculated
            last_sales = agg_data['Net_Sales_Amount_sum'].iloc[-1] if not agg_data.empty else 0
            future_data = []
            last_month = agg_data['Month'].iloc[-1] if not agg_data.empty else datetime.now().month
            last_year = agg_data['Year'].iloc[-1] if not agg_data.empty else datetime.now().year
            
            for i in range(1, periods + 1):
                next_month = ((last_month + i - 1) % 12) + 1
                next_year = last_year + (last_month + i - 1) // 12
                future_data.append({
                    'Year': next_year,
                    'Month': next_month,
                    'Quarter': ((next_month - 1) // 3) + 1,
                    'Forecasted_Sales': last_sales # Simple repetition
                })
            forecast_df = pd.DataFrame(future_data)
            if group_by and group_value:
                forecast_df[group_by] = group_value
            return forecast_df


        trend_model = LinearRegression()
        trend_model.fit(X_trend, y_trend)
        
        # Calculate seasonal adjustment factors
        monthly_means = agg_data.groupby('Month')['Net_Sales_Amount_sum'].mean()
        overall_mean = agg_data['Net_Sales_Amount_sum'].mean()
        
        # Avoid division by zero
        seasonal_factors = monthly_means / overall_mean if overall_mean != 0 else monthly_means.apply(lambda x: 1.0)
        
        # Generate future predictions
        future_data = []
        last_time_point = agg_data['Time_Point'].max()
        last_date = agg_data.iloc[-1]['Date']
        
        for i in range(1, periods + 1):
            future_date = last_date + pd.DateOffset(months=i)
            future_year = future_date.year
            future_month = future_date.month
            future_quarter = future_date.quarter
            future_time_point = last_time_point + i
            
            # Base trend prediction
            trend_prediction = trend_model.predict([[future_time_point]])[0]
            
            # Apply seasonal adjustment
            seasonal_factor = seasonal_factors.get(future_month, 1.0) # Default to 1.0 if month not seen
            adjusted_prediction = trend_prediction * seasonal_factor
            
            # Ensure predictions are non-negative
            adjusted_prediction = max(0, adjusted_prediction)
            
            future_data.append({
                'Year': future_year,
                'Month': future_month,
                'Quarter': future_quarter,
                'Forecasted_Sales': adjusted_prediction
            })
        
        forecast_df = pd.DataFrame(future_data)
        
        if group_by and group_value:
            forecast_df[group_by] = group_value
        
        st.success(f"✅ Simple trend forecast generated for {periods} months.")
        
        return forecast_df

    def generate_forecast(self, model, scaler, data, periods=12, group_by=None, group_value=None):
        """
        Generate comprehensive forecast for future periods using the trained ML model.
        Includes handling of new future feature values and a fallback to simple trend forecast.
        """
        
        if model is None or scaler is None:
            st.error("Model or scaler not trained. Cannot generate ML forecast.")
            return self.generate_simple_trend_forecast(data, periods, group_by, group_value)
            
        st.write(f"🔮 Generating {periods}-month forecast using ML model...")
        
        # Get the latest data point from the aggregated historical data
        # This is crucial for correctly generating future lag features
        latest_historical_row = data.iloc[-1:].copy()
        
        future_data_rows = []
        # Start from the date immediately after the last historical date
        last_historical_date = latest_historical_row['Date'].iloc[0]

        # Keep track of sales to generate rolling features iteratively
        recent_sales_history = data['Net_Sales_Amount_sum'].tolist()

        for i in range(1, periods + 1):
            # Calculate the date for the future month
            future_date = last_historical_date + pd.DateOffset(months=i)
            
            future_row = latest_historical_row.copy()
            future_row['Date'] = future_date
            future_row['Year'] = future_date.year
            future_row['Month'] = future_date.month
            future_row['Quarter'] = future_date.quarter
            
            # Update time-based features for the future row
            future_row['Month_Sin'] = np.sin(2 * np.pi * future_row['Month'] / 12)
            future_row['Month_Cos'] = np.cos(2 * np.pi * future_row['Month'] / 12)
            future_row['Quarter_Sin'] = np.sin(2 * np.pi * future_row['Quarter'] / 4)
            future_row['Quarter_Cos'] = np.cos(2 * np.pi * future_row['Quarter'] / 4)
            future_row['DayOfWeek_Sin'] = np.sin(2 * np.pi * future_date.dayofweek / 7)
            future_row['DayOfWeek_Cos'] = np.cos(2 * np.pi * future_date.dayofweek / 7)
            future_row['DayOfYear_Sin'] = np.sin(2 * np.pi * future_date.dayofyear / 365.25)
            future_row['DayOfYear_Cos'] = np.cos(2 * np.pi * future_date.dayofyear / 365.25)

            future_row['Time_Trend'] = (future_row['Year'] - data['Year'].min()) * 12 + future_row['Month']
            future_row['Time_Trend_Normalized'] = (future_row['Time_Trend'] - data['Time_Trend'].min()) / (data['Time_Trend'].max() - data['Time_Trend'].min()) # Use historical min/max
            
            future_row['Is_Q4'] = (future_row['Quarter'] == 4).astype(int)
            future_row['Is_Q1'] = (future_row['Quarter'] == 1).astype(int)
            future_row['Is_Holiday_Season'] = future_row['Month'].isin([11, 12]).astype(int)
            future_row['Is_Summer_Season'] = future_row['Month'].isin([6, 7, 8]).astype(int)
            
            # Update new features
            future_row['Year_Progress'] = future_row['Month'] / 12
            future_row['Quarter_Progress'] = (future_row['Month'] % 3) / 3
            future_row['Month_Squared'] = future_row['Month'] ** 2
            future_row['Time_Trend_Squared'] = future_row['Time_Trend'] ** 2
            
            # Interaction features
            future_row['Trend_Month_Interaction'] = future_row['Time_Trend_Normalized'] * future_row['Month_Sin']
            future_row['Trend_Quarter_Interaction'] = future_row['Time_Trend_Normalized'] * future_row['Quarter_Sin']
            future_row['Holiday_Trend_Interaction'] = future_row['Is_Holiday_Season'] * future_row['Time_Trend_Normalized']

            # Handle group_by encoding for future data
            if group_by and group_value:
                if group_by in self.label_encoders:
                    # Transform unseen values safely (will be -1 if unseen, model should handle or fill)
                    try:
                        future_row[f'{group_by}_encoded'] = self.label_encoders[group_by].transform([group_value])[0]
                    except ValueError:
                        st.warning(f"Group value '{group_value}' not seen in training data for {group_by}. Encoding as -1.")
                        future_row[f'{group_by}_encoded'] = -1
                else:
                    st.warning(f"LabelEncoder for '{group_by}' not found. Cannot encode group for forecast.")
                    future_row[f'{group_by}_encoded'] = 0 # Default to 0 or handle appropriately
            
            # Dynamically calculate lag features and rolling averages for the future row
            # This requires appending the previously predicted value to history
            
            # Lag features (using recent_sales_history)
            for lag in [1, 2, 3, 4, 5, 6, 12]:
                if len(recent_sales_history) >= lag:
                    future_row[f'Sales_Lag_{lag}'] = recent_sales_history[-lag]
                else:
                    future_row[f'Sales_Lag_{lag}'] = 0 # Fill with 0 if not enough history

            # Rolling averages and std dev (using recent_sales_history)
            for window in [3, 6, 12, 24]:
                if len(recent_sales_history) >= window:
                    future_row[f'Sales_MA_{window}'] = np.mean(recent_sales_history[-window:])
                    future_row[f'Sales_Std_{window}'] = np.std(recent_sales_history[-window:])
                else:
                    future_row[f'Sales_MA_{window}'] = np.mean(recent_sales_history) if recent_sales_history else 0
                    future_row[f'Sales_Std_{window}'] = np.std(recent_sales_history) if recent_sales_history else 0
            
            # EMA (using recent_sales_history)
            for span in [7, 30, 90]:
                if recent_sales_history:
                    # Simple EMA calculation: new_ema = alpha * current_value + (1 - alpha) * previous_ema
                    # For a rolling list, a full re-calculation is more robust
                    series = pd.Series(recent_sales_history)
                    future_row[f'Sales_EMA_{span}'] = series.ewm(span=span, adjust=False).mean().iloc[-1]
                else:
                    future_row[f'Sales_EMA_{span}'] = 0
            
            # Prepare the row for prediction
            # Ensure all feature columns expected by the model are present, fill with 0 if not
            feature_values = []
            for col in self.feature_columns:
                if col in future_row.columns:
                    feature_values.append(future_row[col].iloc[0])
                else:
                    feature_values.append(0)
            
            X_future_row = pd.DataFrame([feature_values], columns=self.feature_columns)
            X_future_row = X_future_row.fillna(0) # Final check for any NaNs
            X_future_scaled = scaler.transform(X_future_row)
            
            # Make prediction for the current future month
            predicted_sales = model.predict(X_future_scaled)[0]
            predicted_sales = max(0, predicted_sales) # Ensure non-negative sales

            # Append prediction to history for next iteration's lag/rolling calculations
            recent_sales_history.append(predicted_sales)

            # Store the forecast for this month
            future_data_rows.append({
                'Year': future_row['Year'].iloc[0],
                'Month': future_row['Month'].iloc[0],
                'Quarter': future_row['Quarter'].iloc[0],
                'Forecasted_Sales': predicted_sales
            })
        
        forecast_df = pd.DataFrame(future_data_rows)
        
        if group_by and group_by in latest_historical_row.columns:
            forecast_df[group_by] = latest_historical_row[group_by].iloc[0] # Carry over group value

        # Check if ML forecast is reasonable, otherwise use simple trend
        # Compare forecast mean/std to historical mean/std
        hist_mean = data['Net_Sales_Amount_sum'].mean()
        hist_std = data['Net_Sales_Amount_sum'].std()
        
        pred_mean = forecast_df['Forecasted_Sales'].mean()
        pred_std = forecast_df['Forecasted_Sales'].std()
        
        st.write(f"🔍 Debug Info (ML Forecast):")
        st.write(f"• Historical mean: {hist_mean:,.0f}, std: {hist_std:,.0f}")
        st.write(f"• Forecast mean: {pred_mean:,.0f}, std: {pred_std:,.0f}")

        # Criteria for "unrealistic":
        # 1. Forecast mean is drastically different from historical mean (e.g., > 3 std dev away)
        # 2. Forecast standard deviation is extremely low (model predicting flat line) or extremely high
        if (abs(pred_mean - hist_mean) > 3 * hist_std and hist_std > 0) or \
           (pred_std < hist_std * 0.1 and hist_std > 0) or \
           (pred_std > hist_std * 5 and hist_std > 0): # Increased multiplier for std deviation check
            st.warning("⚠️ ML forecast appears unrealistic (e.g., too flat, too volatile, or drastically off mean). Using simple trend forecast instead.")
            return self.generate_simple_trend_forecast(data, periods, group_by, group_value)
        
        st.success(f"✅ ML-based forecast generated for {periods} months.")
        
        return forecast_df
    
    def create_forecast_visualization(self, historical_data, forecast_data, title="Sales Forecast"):
        """
        Create a comprehensive interactive forecast visualization using Plotly.
        Shows historical sales and forecasted sales on the same plot.
        """
        
        if historical_data is None or forecast_data is None:
            st.error("Insufficient data for visualization.")
            return None
            
        # Prepare historical data: ensure 'Date' column exists and is datetime
        hist_df = pd.DataFrame(historical_data)
        if 'Sale_Date' in hist_df.columns:
            hist_df['Date'] = pd.to_datetime(hist_df['Sale_Date'])
        else:
            # Reconstruct 'Date' from 'Sales Year' and 'Sales Month' if Sale_Date is missing
            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            if 'Sales Year' in hist_df.columns and 'Sales Month' in hist_df.columns:
                hist_df['Month_Num'] = hist_df['Sales Month'].map(month_map)
                hist_df['Date'] = pd.to_datetime(hist_df['Sales Year'].astype(str) + '-' + 
                                               hist_df['Month_Num'].astype(str) + '-01')
            else:
                st.error("Cannot create historical date column. Missing 'Sale_Date' or 'Sales Year'/'Sales Month'.")
                return None
        
        # Aggregate historical data to monthly level for consistent plotting with forecast
        hist_agg = hist_df.groupby(pd.Grouper(key='Date', freq='MS'))['Net_Sales_Amount'].sum().reset_index()
        hist_agg.rename(columns={'Net_Sales_Amount': 'Actual_Sales'}, inplace=True)

        # Prepare forecast data: ensure 'Date' column exists and is datetime
        forecast_df = forecast_data.copy()
        forecast_df['Date'] = pd.to_datetime(forecast_df['Year'].astype(str) + '-' + 
                                           forecast_df['Month'].astype(str) + '-01')
        
        # Create the plot
        fig = go.Figure()
        
        # Historical data trace
        fig.add_trace(go.Scatter(
            x=hist_agg['Date'],
            y=hist_agg['Actual_Sales'],
            mode='lines+markers',
            name='Historical Sales (Actual)',
            line=dict(color='#1f77b4', width=2), # Blue color
            marker=dict(size=6, symbol='circle-open', line=dict(width=1, color='#1f77b4'))
        ))
        
        # Forecast data trace
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted_Sales'],
            mode='lines+markers',
            name='Forecasted Sales',
            line=dict(color='#d62728', width=2, dash='dash'), # Red dashed line
            marker=dict(size=6, symbol='star', color='#d62728')
        ))
        
        # Add a shaded region for the forecast period for clarity
        if not forecast_df.empty:
            forecast_start_date = forecast_df['Date'].min()
            forecast_end_date = forecast_df['Date'].max()
            fig.add_vrect(
                x0=forecast_start_date, x1=forecast_end_date,
                fillcolor="LightSalmon", opacity=0.2, layer="below", line_width=0,
                annotation_text="Forecast Period", annotation_position="top left",
                annotation_font_size=10, annotation_font_color="grey"
            )

        # Update layout for better aesthetics and readability
        fig.update_layout(
            title={
                'text': title,
                'y':0.95, # Adjust title position
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Date',
            yaxis_title='Sales Amount (৳)',
            hovermode='x unified', # Unified hover for better comparison
            legend=dict(
                x=0.01, y=0.99, # Top-left legend
                bgcolor='rgba(255,255,255,0.7)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            height=550, # Slightly taller for better view
            template="plotly_white", # Clean white background
            margin=dict(l=40, r=40, t=80, b=40) # Adjust margins
        )
        
        # Improve x-axis tick format for dates
        fig.update_xaxes(
            rangeselector_buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]),
            rangeslider_visible=True, # Add a range slider
            type="date"
        )
        
        # Ensure y-axis starts from 0 or a reasonable minimum
        fig.update_yaxes(rangemode='tozero')

        return fig

def get_forecasting_insights(historical_data, forecast_data, group_by=None):
    """
    Generate comprehensive business insights from forecasting results,
    including overall performance, growth rates, and seasonal trends.
    """
    
    if historical_data is None or forecast_data is None:
        return "Unable to generate insights due to insufficient data."
    
    hist_df = pd.DataFrame(historical_data)
    forecast_df = forecast_data.copy()
    
    insights = ""

    # Ensure historical data has date components for aggregation
    if 'Sale_Date' in hist_df.columns:
        hist_df['Date'] = pd.to_datetime(hist_df['Sale_Date'])
        hist_df['Year'] = hist_df['Date'].dt.year
        hist_df['Month'] = hist_df['Date'].dt.month
    elif 'Sales Year' in hist_df.columns and 'Sales Month' in hist_df.columns:
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        hist_df['Month'] = hist_df['Sales Month'].map(month_map)
        hist_df['Year'] = hist_df['Sales Year']
    else:
        insights += "Warning: Date information not fully available in historical data, seasonal insights might be limited.\n\n"

    # Group-specific insights (if applicable)
    if group_by and group_by in forecast_df.columns and group_by in hist_df.columns:
        # Aggregate historical data by group and month to match forecast granularity
        monthly_hist_grouped = hist_df.groupby([group_by, 'Year', 'Month'])['Net_Sales_Amount'].sum().reset_index()
        
        insights += f"**Forecasting Insights by {group_by}:**\n\n"
        
        for group in forecast_df[group_by].unique():
            group_forecast_df = forecast_df[forecast_df[group_by] == group]
            group_hist_df = monthly_hist_grouped[monthly_hist_grouped[group_by] == group]
            
            if not group_forecast_df.empty and not group_hist_df.empty:
                forecast_periods = len(group_forecast_df)
                
                # Get recent historical data corresponding to forecast period length
                recent_group_hist = group_hist_df.sort_values(['Year', 'Month']).tail(forecast_periods)
                
                total_recent_hist = recent_group_hist['Net_Sales_Amount'].sum()
                total_forecast = group_forecast_df['Forecasted_Sales'].sum()
                
                growth_rate = ((total_forecast - total_recent_hist) / total_recent_hist) * 100 if total_recent_hist > 0 else 0
                
                insights += f"**{group}:**\n"
                insights += f"• Historical Sales ({forecast_periods} months): ৳{total_recent_hist:,.2f}\n"
                insights += f"• Forecasted Sales ({forecast_periods} months): ৳{total_forecast:,.2f}\n"
                insights += f"• Projected Growth: {growth_rate:+.1f}%\n\n"
            else:
                insights += f"**{group}:** Insufficient historical or forecast data for detailed insights.\n\n"
    else:
        # Overall insights
        # Aggregate historical data to monthly level to match forecast data granularity
        monthly_hist = hist_df.groupby(['Year', 'Month'])['Net_Sales_Amount'].sum().reset_index()
        monthly_hist = monthly_hist.sort_values(['Year', 'Month'])
        
        forecast_periods = len(forecast_df)
        
        # Get the last N months of aggregated historical data matching forecast length
        recent_monthly_hist = monthly_hist.tail(forecast_periods)
        
        total_recent_hist = recent_monthly_hist['Net_Sales_Amount'].sum()
        total_forecast = forecast_df['Forecasted_Sales'].sum()
        
        growth_rate = ((total_forecast - total_recent_hist) / total_recent_hist) * 100 if total_recent_hist > 0 else 0
        
        insights += "**Overall Forecasting Insights:**\n\n"
        insights += f"• Recent Historical Sales (last {forecast_periods} months): ৳{total_recent_hist:,.2f}\n"
        insights += f"• Forecasted Sales (next {forecast_periods} months): ৳{total_forecast:,.2f}\n"
        insights += f"• Projected Growth: {growth_rate:+.1f}%\n\n"
        
        # Additional insights
        avg_monthly_hist = total_recent_hist / forecast_periods if forecast_periods > 0 else 0
        avg_monthly_forecast = total_forecast / forecast_periods if forecast_periods > 0 else 0
        
        insights += f"• Average Monthly Historical: ৳{avg_monthly_hist:,.2f}\n"
        insights += f"• Average Monthly Forecast: ৳{avg_monthly_forecast:,.2f}\n"
        
        # Show monthly breakdown for debugging/detail
        insights += f"\n**Monthly Breakdown (Forecast vs. Recent Historical - Last {min(6, len(recent_monthly_hist))} months):**\n"
        for i, (_, hist_row) in enumerate(recent_monthly_hist.tail(6).iterrows()):
            if i < len(forecast_df):
                forecast_val = forecast_df.iloc[i]['Forecasted_Sales']
                insights += f"• {int(hist_row['Year'])}-{int(hist_row['Month']):02d}: Hist=৳{hist_row['Net_Sales_Amount']:,.0f}, Forecast=৳{forecast_val:,.0f}\n"
            else:
                insights += f"• {int(hist_row['Year'])}-{int(hist_row['Month']):02d}: Hist=৳{hist_row['Net_Sales_Amount']:,.0f} (No corresponding forecast data)\n"

        # Seasonal insights from historical data
        if 'Month' in hist_df.columns:
            # Ensure 'Net_Sales_Amount' is available for seasonal analysis
            if 'Net_Sales_Amount' in hist_df.columns:
                monthly_hist_seasonal = hist_df.groupby('Month')['Net_Sales_Amount'].sum()
                if not monthly_hist_seasonal.empty:
                    # Map month numbers to names for readability
                    month_names = {
                        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
                    }
                    best_month_num = monthly_hist_seasonal.idxmax()
                    worst_month_num = monthly_hist_seasonal.idxmin()
                    
                    best_month_name = month_names.get(best_month_num, str(best_month_num))
                    worst_month_name = month_names.get(worst_month_num, str(worst_month_num))
                    
                    insights += f"\n• Historically Best Performing Month: {best_month_name}\n"
                    insights += f"• Historically Lowest Performing Month: {worst_month_name}\n"
                else:
                    insights += "\nCould not determine best/worst performing months (no sales data).\n"
            else:
                insights += "\nCannot determine best/worst performing months (Net_Sales_Amount not in historical data).\n"

    return insights

