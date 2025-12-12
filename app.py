import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.impute import SimpleImputer
import traceback
import warnings
import datetime
import math
from scipy import stats
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'ml-preprocessor-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Create directories if not exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def detect_problem_type(df, target_column):
    """Detect if problem is classification or regression"""
    if target_column not in df.columns:
        return 'unknown'
    
    # Check if target column exists
    if df[target_column].isnull().all():
        return 'unknown'
    
    unique_values = df[target_column].nunique()
    
    if unique_values <= 15 or df[target_column].dtype == 'object':
        return 'classification'
    else:
        if pd.api.types.is_numeric_dtype(df[target_column]):
            return 'regression'
        else:
            return 'classification'

def detect_column_types(df):
    """Detect categorical and numerical columns"""
    categorical_cols = []
    numerical_cols = []
    datetime_cols = []
    
    for col in df.columns:
        # Skip if all values are null
        if df[col].isnull().all():
            continue
            
        # Try to detect datetime columns
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                continue
        except:
            pass
            
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # Check if it might be numeric
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.8:
                    numerical_cols.append(col)
                else:
                    categorical_cols.append(col)
            except:
                categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)
    
    return {
        'categorical': categorical_cols,
        'numerical': numerical_cols,
        'datetime': datetime_cols
    }

def load_dataset(filepath):
    """Load dataset from various formats"""
    try:
        if filepath.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
                    # Try to parse datetime columns
                    for col in df.select_dtypes(include=['object']).columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='ignore')
                        except:
                            pass
                    return df
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with encoding {encoding}: {e}")
                    continue
            # If all encodings fail, try without specifying encoding
            return pd.read_csv(filepath, low_memory=False)
            
        elif filepath.endswith(('.xlsx', '.xls')):
            try:
                return pd.read_excel(filepath, engine='openpyxl')
            except:
                try:
                    return pd.read_excel(filepath)
                except:
                    return None
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        traceback.print_exc()
        return None

def calculate_column_statistics(df, column):
    """Calculate detailed statistics for a column"""
    stats = {
        'name': column,
        'dtype': str(df[column].dtype),
        'total_count': int(len(df)),
        'non_null_count': int(df[column].notna().sum()),
        'null_count': int(df[column].isna().sum()),
        'null_percentage': float((df[column].isna().sum() / len(df)) * 100),
        'unique_count': int(df[column].nunique())
    }
    
    # For numeric columns
    if pd.api.types.is_numeric_dtype(df[column]):
        numeric_series = pd.to_numeric(df[column], errors='coerce')
        non_null_values = numeric_series.dropna()
        
        if len(non_null_values) > 0:
            stats['min'] = float(non_null_values.min())
            stats['max'] = float(non_null_values.max())
            stats['mean'] = float(non_null_values.mean())
            stats['median'] = float(non_null_values.median())
            stats['std'] = float(non_null_values.std())
            stats['variance'] = float(non_null_values.var())
            stats['skewness'] = float(non_null_values.skew()) if len(non_null_values) > 2 else None
            stats['kurtosis'] = float(non_null_values.kurtosis()) if len(non_null_values) > 3 else None
            
            # Percentiles
            percentiles = non_null_values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
            stats['percentiles'] = {
                'p1': float(percentiles.get(0.01, 0)),
                'p5': float(percentiles.get(0.05, 0)),
                'p25': float(percentiles.get(0.25, 0)),
                'p50': float(percentiles.get(0.5, 0)),
                'p75': float(percentiles.get(0.75, 0)),
                'p95': float(percentiles.get(0.95, 0)),
                'p99': float(percentiles.get(0.99, 0))
            }
            
            # Outliers using IQR
            Q1 = non_null_values.quantile(0.25)
            Q3 = non_null_values.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = non_null_values[(non_null_values < lower_bound) | (non_null_values > upper_bound)]
                stats['outlier_count'] = int(len(outliers))
                stats['outlier_percentage'] = float((len(outliers) / len(non_null_values)) * 100)
    
    # For categorical columns
    elif df[column].dtype == 'object' or df[column].dtype.name == 'category':
        value_counts = df[column].value_counts(dropna=True)
        stats['most_frequent'] = str(value_counts.index[0]) if len(value_counts) > 0 else None
        stats['most_frequent_count'] = int(value_counts.iloc[0]) if len(value_counts) > 0 else None
        stats['least_frequent'] = str(value_counts.index[-1]) if len(value_counts) > 0 else None
        stats['least_frequent_count'] = int(value_counts.iloc[-1]) if len(value_counts) > 0 else None
        
        # Top 10 values
        top_values = value_counts.head(10)
        stats['top_values'] = {str(k): int(v) for k, v in top_values.items()}
    
    # For datetime columns
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        datetime_series = pd.to_datetime(df[column], errors='coerce')
        non_null_dates = datetime_series.dropna()
        
        if len(non_null_dates) > 0:
            stats['min_date'] = str(non_null_dates.min())
            stats['max_date'] = str(non_null_dates.max())
            stats['date_range_days'] = int((non_null_dates.max() - non_null_dates.min()).days)
    
    return stats

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(filepath)
            
            df = load_dataset(filepath)
            if df is None or df.empty:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({'error': 'Failed to read the file or file is empty.'}), 400
            
            print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Clean column names (remove spaces, special characters)
            df.columns = [str(col).strip().replace(' ', '_').replace('.', '_').replace('-', '_') 
                         for col in df.columns]
            
            column_types = detect_column_types(df)
            
            # Optimize data types
            for col in df.columns:
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            missing_values_sum = df.isnull().sum()
            missing_percentage = (missing_values_sum / len(df) * 100)
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            
            dataset_info = {
                'filename': filename,
                'rows': int(len(df)),
                'columns': int(len(df.columns)),
                'column_names': [str(col) for col in df.columns.tolist()],
                'dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
                'missing_values': {str(col): int(val) for col, val in missing_values_sum.items()},
                'missing_percentage': {str(col): float(val) for col, val in missing_percentage.items()},
                'duplicates': int(df.duplicated().sum()),
                'column_types': column_types,
                'memory_usage_mb': float(memory_usage),
                'preview': df.head(10).replace({np.nan: None}).to_dict('records')
            }
            
            # Convert preview data to Python native types
            for record in dataset_info['preview']:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.int64)):
                        record[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        record[key] = float(value)
                    elif pd.isna(value):
                        record[key] = None
                    elif isinstance(value, (pd.Timestamp, datetime.datetime)):
                        record[key] = value.isoformat() if pd.notna(value) else None
            
            original_filename = 'original_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + filename
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            df.to_csv(original_path, index=False)
            
            response_data = {
                'success': True,
                'dataset_info': numpy_to_python(dataset_info),
                'original_file': original_filename,
                'message': f'Successfully uploaded {filename} with {len(df):,} rows and {len(df.columns)} columns'
            }
            
            return jsonify(response_data)
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            
            print(f"Error in upload_file: {str(e)}")
            traceback.print_exc()
            
            return jsonify({'error': f'Failed to process file: {str(e)}'}), 500
    
    return jsonify({'error': 'File type not allowed. Please upload CSV or Excel files only.'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_dataset():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(filepath)
        
        analysis_report = {
            'basic_stats': {},
            'column_analysis': [],
            'correlation_matrix': None,
            'data_quality_metrics': {},
            'column_distributions': {},
            'target_analysis': None
        }
        
        # Calculate data quality metrics
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        analysis_report['data_quality_metrics'] = {
            'total_cells': int(total_cells),
            'missing_cells': int(missing_cells),
            'missing_percentage': float((missing_cells / total_cells) * 100),
            'duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': float((duplicate_rows / len(df)) * 100),
            'columns_with_missing': int((df.isnull().sum() > 0).sum()),
            'columns_high_missing': int(((df.isnull().sum() / len(df)) > 0.5).sum()),
            'zero_variance_columns': int((df.nunique() == 1).sum()),
            'constant_columns': int((df.nunique() <= 1).sum())
        }
        
        # Calculate column-wise statistics
        for column in df.columns:
            try:
                col_stats = calculate_column_statistics(df, column)
                analysis_report['column_analysis'].append(col_stats)
            except Exception as e:
                print(f"Error analyzing column {column}: {e}")
                # Add basic info even if detailed analysis fails
                analysis_report['column_analysis'].append({
                    'name': column,
                    'dtype': str(df[column].dtype),
                    'total_count': int(len(df)),
                    'non_null_count': int(df[column].notna().sum()),
                    'null_count': int(df[column].isna().sum()),
                    'null_percentage': float((df[column].isna().sum() / len(df)) * 100),
                    'unique_count': int(df[column].nunique())
                })
        
        # Calculate basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            basic_stats = df[numeric_cols].describe(percentiles=[.25, .5, .75])
            analysis_report['basic_stats'] = {
                str(col): {
                    str(stat): float(val) if not pd.isna(val) else None 
                    for stat, val in stats.items()
                }
                for col, stats in basic_stats.to_dict().items()
            }
        
        # Calculate correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            try:
                # Drop columns with all NaN values
                numeric_df = df[numeric_cols].dropna(axis=1, how='all')
                if len(numeric_df.columns) > 1:
                    correlation = numeric_df.corr()
                    if not correlation.empty:
                        corr_records = []
                        for i, col1 in enumerate(correlation.columns):
                            for j, col2 in enumerate(correlation.columns):
                                if i < j:
                                    corr_value = correlation.iloc[i, j]
                                    if not pd.isna(corr_value):
                                        corr_records.append({
                                            'column1': str(col1),
                                            'column2': str(col2),
                                            'correlation': float(corr_value),
                                            'abs_correlation': abs(float(corr_value))
                                        })
                        # Sort by absolute correlation
                        corr_records.sort(key=lambda x: x['abs_correlation'], reverse=True)
                        analysis_report['correlation_matrix'] = corr_records[:50]  # Limit to top 50
            except Exception as e:
                print(f"Error calculating correlation: {e}")
                analysis_report['correlation_matrix'] = []
        
        return jsonify({
            'success': True,
            'analysis': numpy_to_python(analysis_report)
        })
        
    except Exception as e:
        print(f"Error in analyze_dataset: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get_column_types', methods=['POST'])
def get_column_types():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        df = pd.read_csv(filepath)
        
        column_types = detect_column_types(df)
        
        suggestions = {
            'encoding_suggestions': [],
            'dtype_conversion_suggestions': [],
            'outlier_suggestions': [],
            'missing_value_suggestions': [],
            'column_removal_suggestions': []
        }
        
        # Encoding suggestions for categorical columns
        for col in column_types['categorical']:
            if col in df.columns:
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                total_count = len(df[col])
                
                # Skip columns with too many missing values
                if null_count / total_count > 0.8:
                    suggestions['column_removal_suggestions'].append({
                        'column': col,
                        'reason': f'High missing values ({null_count/total_count:.1%})',
                        'suggestion': 'Consider removing'
                    })
                elif 2 <= unique_count <= 50:
                    suggestions['encoding_suggestions'].append({
                        'column': col,
                        'unique_values': int(unique_count),
                        'suggested_method': 'label' if unique_count <= 15 else 'onehot',
                        'null_count': int(null_count)
                    })
        
        # Data type conversion suggestions
        for col in column_types['categorical']:
            if col in df.columns:
                try:
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    numeric_count = numeric_series.notna().sum()
                    if numeric_count > len(df) * 0.7:
                        suggestions['dtype_conversion_suggestions'].append({
                            'column': col,
                            'current_dtype': str(df[col].dtype),
                            'suggested_dtype': 'numeric',
                            'conversion_rate': float(numeric_count / len(df))
                        })
                except:
                    pass
        
        # Outlier suggestions for numeric columns
        for col in column_types['numerical']:
            if col in df.columns and df[col].notna().sum() > 10:  # Need at least 10 values
                numeric_series = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_series) > 0:
                    q1 = numeric_series.quantile(0.25)
                    q3 = numeric_series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outlier_count = ((numeric_series < lower_bound) | (numeric_series > upper_bound)).sum()
                        if outlier_count > 0:
                            suggestions['outlier_suggestions'].append({
                                'column': col,
                                'outlier_count': int(outlier_count),
                                'outlier_percentage': float((outlier_count / len(numeric_series)) * 100),
                                'method': 'iqr'
                            })
        
        # Missing value suggestions
        for col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                
                if null_percentage > 50:
                    action = 'remove'
                elif null_percentage > 20:
                    action = 'impute_advanced'
                else:
                    action = 'impute_simple'
                
                suggestions['missing_value_suggestions'].append({
                    'column': col,
                    'null_count': int(null_count),
                    'null_percentage': float(null_percentage),
                    'suggested_action': action
                })
        
        # Column removal suggestions
        for col in df.columns:
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            
            # Check for constant columns
            if unique_count <= 1:
                suggestions['column_removal_suggestions'].append({
                    'column': col,
                    'reason': 'Constant or single value',
                    'suggestion': 'Remove - no predictive value'
                })
            # Check for high cardinality with low frequency
            elif unique_count > len(df) * 0.9 and col in column_types['categorical']:
                suggestions['column_removal_suggestions'].append({
                    'column': col,
                    'reason': f'High cardinality ({unique_count} unique values)',
                    'suggestion': 'Consider removing or hashing'
                })
        
        return jsonify({
            'success': True,
            'column_types': column_types,
            'suggestions': suggestions
        })
        
    except Exception as e:
        print(f"Error in get_column_types: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_dataset():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        preprocessing_steps = data.get('steps', [])
        target_column = data.get('target_column')
        columns_to_remove = data.get('columns_to_remove', [])
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original file not found'}), 404
        
        df = pd.read_csv(original_path)
        original_shape = df.shape
        
        df_processed = df.copy()
        preprocessing_report = []
        performance_metrics = {
            'rows_removed': 0,
            'columns_removed': 0,
            'missing_values_filled': 0,
            'outliers_removed': 0,
            'encoding_applied': 0,
            'dtype_changes': 0,
            'columns_dropped': []
        }
        
        # Step 1: Remove specified columns
        if columns_to_remove:
            columns_to_drop = [col for col in columns_to_remove if col in df_processed.columns]
            if columns_to_drop:
                df_processed.drop(columns=columns_to_drop, inplace=True)
                performance_metrics['columns_removed'] += len(columns_to_drop)
                performance_metrics['columns_dropped'] = columns_to_drop
                preprocessing_report.append({
                    'step': 'Removed user-selected columns',
                    'details': f'Columns removed: {columns_to_drop}',
                    'type': 'column_removal',
                    'status': 'success'
                })
        
        # Step 2: Apply preprocessing steps
        for step in preprocessing_steps:
            try:
                step_type = step.get('type')
                
                if step_type == 'drop_high_missing':
                    threshold = step.get('threshold', 50)
                    missing_percentage = (df_processed.isnull().sum() / len(df_processed)) * 100
                    columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
                    if columns_to_drop:
                        df_processed.drop(columns=columns_to_drop, inplace=True)
                        performance_metrics['columns_removed'] += len(columns_to_drop)
                        preprocessing_report.append({
                            'step': f'Dropped columns with >{threshold}% missing values',
                            'details': f'Columns removed: {columns_to_drop}',
                            'type': 'missing_values',
                            'status': 'success'
                        })
                
                elif step_type == 'remove_duplicates':
                    before = len(df_processed)
                    df_processed.drop_duplicates(inplace=True)
                    after = len(df_processed)
                    removed = before - after
                    if removed > 0:
                        performance_metrics['rows_removed'] += removed
                        preprocessing_report.append({
                            'step': 'Removed duplicate rows',
                            'details': f'Removed {removed} duplicate rows',
                            'type': 'duplicates',
                            'status': 'success'
                        })
                
                elif step_type == 'change_dtype':
                    column = step.get('column')
                    new_dtype = step.get('dtype')
                    
                    if column and column in df_processed.columns and new_dtype:
                        old_dtype = str(df_processed[column].dtype)
                        
                        try:
                            if new_dtype == 'numeric':
                                df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
                            elif new_dtype == 'datetime':
                                df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
                            elif new_dtype == 'category':
                                df_processed[column] = df_processed[column].astype('category')
                            elif new_dtype == 'string':
                                df_processed[column] = df_processed[column].astype('str')
                            
                            performance_metrics['dtype_changes'] += 1
                            preprocessing_report.append({
                                'step': f'Changed data type of column "{column}"',
                                'details': f'From {old_dtype} to {new_dtype}',
                                'type': 'dtype_conversion',
                                'status': 'success'
                            })
                        except Exception as e:
                            preprocessing_report.append({
                                'step': f'Failed to change data type of column "{column}"',
                                'details': str(e),
                                'type': 'dtype_conversion',
                                'status': 'error'
                            })
                
                elif step_type == 'encoding':
                    column = step.get('column')
                    method = step.get('method')
                    
                    if column and column in df_processed.columns and method:
                        if method == 'label':
                            try:
                                # Handle NaN values before encoding
                                df_processed[column] = df_processed[column].fillna('MISSING')
                                le = LabelEncoder()
                                df_processed[column] = le.fit_transform(df_processed[column].astype(str))
                                performance_metrics['encoding_applied'] += 1
                                preprocessing_report.append({
                                    'step': f'Applied Label Encoding to column "{column}"',
                                    'details': f'Number of unique values: {len(le.classes_)}',
                                    'type': 'encoding',
                                    'status': 'success'
                                })
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to apply Label Encoding to column "{column}"',
                                    'details': str(e),
                                    'type': 'encoding',
                                    'status': 'error'
                                })
                        elif method == 'onehot':
                            try:
                                # Handle NaN values
                                df_processed[column] = df_processed[column].fillna('MISSING')
                                dummies = pd.get_dummies(df_processed[column], prefix=column)
                                df_processed = pd.concat([df_processed.drop(column, axis=1), dummies], axis=1)
                                performance_metrics['encoding_applied'] += 1
                                preprocessing_report.append({
                                    'step': f'Applied One-Hot Encoding to column "{column}"',
                                    'details': f'Created {len(dummies.columns)} new columns',
                                    'type': 'encoding',
                                    'status': 'success'
                                })
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to apply One-Hot Encoding to column "{column}"',
                                    'details': str(e),
                                    'type': 'encoding',
                                    'status': 'error'
                                })
                
                elif step_type == 'handle_missing':
                    column = step.get('column')
                    method = step.get('method')
                    custom_value = step.get('custom_value')
                    
                    if column and column in df_processed.columns and method:
                        missing_count = df_processed[column].isnull().sum()
                        
                        if method == 'drop':
                            before = len(df_processed)
                            df_processed = df_processed.dropna(subset=[column])
                            after = len(df_processed)
                            removed = before - after
                            performance_metrics['rows_removed'] += removed
                            preprocessing_report.append({
                                'step': f'Dropped rows with missing values in column "{column}"',
                                'details': f'Rows dropped: {removed}, Rows remaining: {after}',
                                'type': 'missing_values',
                                'status': 'success'
                            })
                        else:
                            if method == 'mean':
                                fill_value = df_processed[column].mean()
                            elif method == 'median':
                                fill_value = df_processed[column].median()
                            elif method == 'mode':
                                mode_values = df_processed[column].mode()
                                fill_value = mode_values[0] if not mode_values.empty else 0
                            elif method == 'forward_fill':
                                df_processed[column].fillna(method='ffill', inplace=True)
                                fill_value = 'forward fill'
                            elif method == 'backward_fill':
                                df_processed[column].fillna(method='bfill', inplace=True)
                                fill_value = 'backward fill'
                            elif method == 'interpolate':
                                df_processed[column].interpolate(method='linear', inplace=True)
                                fill_value = 'linear interpolation'
                            elif method == 'custom' and custom_value is not None:
                                fill_value = custom_value
                            else:
                                fill_value = 0
                            
                            if method not in ['forward_fill', 'backward_fill', 'interpolate']:
                                df_processed[column].fillna(fill_value, inplace=True)
                            
                            performance_metrics['missing_values_filled'] += int(missing_count)
                            preprocessing_report.append({
                                'step': f'Filled missing values in column "{column}"',
                                'details': f'Method: {method}, Rows filled: {int(missing_count)}',
                                'type': 'missing_values',
                                'status': 'success'
                            })
                
                elif step_type == 'remove_outliers':
                    column = step.get('column')
                    method = step.get('method', 'iqr')
                    threshold = step.get('threshold', 1.5)
                    
                    if column and column in df_processed.columns:
                        if pd.api.types.is_numeric_dtype(df_processed[column]):
                            try:
                                numeric_series = pd.to_numeric(df_processed[column], errors='coerce')
                                
                                if method == 'iqr':
                                    Q1 = numeric_series.quantile(0.25)
                                    Q3 = numeric_series.quantile(0.75)
                                    IQR = Q3 - Q1
                                    
                                    if IQR > 0:
                                        lower_bound = Q1 - threshold * IQR
                                        upper_bound = Q3 + threshold * IQR
                                        
                                        before = len(df_processed)
                                        mask = (numeric_series >= lower_bound) & (numeric_series <= upper_bound)
                                        df_processed = df_processed[mask]
                                        after = len(df_processed)
                                        removed = before - after
                                        
                                        if removed > 0:
                                            performance_metrics['outliers_removed'] += removed
                                            preprocessing_report.append({
                                                'step': f'Removed outliers from column "{column}"',
                                                'details': f'Removed {removed} rows using IQR method',
                                                'type': 'outliers',
                                                'status': 'success'
                                            })
                                
                                elif method == 'zscore':
                                    z_scores = np.abs(stats.zscore(numeric_series.dropna()))
                                    threshold_z = threshold
                                    
                                    before = len(df_processed)
                                    mask = (z_scores < threshold_z) | numeric_series.isna()
                                    df_processed = df_processed[mask.reindex(df_processed.index, fill_value=True)]
                                    after = len(df_processed)
                                    removed = before - after
                                    
                                    if removed > 0:
                                        performance_metrics['outliers_removed'] += removed
                                        preprocessing_report.append({
                                            'step': f'Removed outliers from column "{column}"',
                                            'details': f'Removed {removed} rows using Z-score method',
                                            'type': 'outliers',
                                            'status': 'success'
                                        })
                            
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to remove outliers from column "{column}"',
                                    'details': str(e),
                                    'type': 'outliers',
                                    'status': 'error'
                                })
                
                elif step_type == 'scale_column':
                    column = step.get('column')
                    method = step.get('method', 'standard')
                    
                    if column and column in df_processed.columns:
                        if pd.api.types.is_numeric_dtype(df_processed[column]):
                            try:
                                if method == 'standard':
                                    scaler = StandardScaler()
                                    df_processed[column] = scaler.fit_transform(df_processed[[column]])
                                elif method == 'minmax':
                                    min_val = df_processed[column].min()
                                    max_val = df_processed[column].max()
                                    if max_val > min_val:
                                        df_processed[column] = (df_processed[column] - min_val) / (max_val - min_val)
                                
                                preprocessing_report.append({
                                    'step': f'Scaled column "{column}"',
                                    'details': f'Method: {method} scaling',
                                    'type': 'scaling',
                                    'status': 'success'
                                })
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to scale column "{column}"',
                                    'details': str(e),
                                    'type': 'scaling',
                                    'status': 'error'
                                })
                
                elif step_type == 'batch_encoding':
                    columns = step.get('columns', [])
                    method = step.get('method', 'label')
                    
                    valid_columns = [col for col in columns if col in df_processed.columns]
                    
                    for column in valid_columns:
                        if method == 'label':
                            try:
                                df_processed[column] = df_processed[column].fillna('MISSING')
                                le = LabelEncoder()
                                df_processed[column] = le.fit_transform(df_processed[column].astype(str))
                                performance_metrics['encoding_applied'] += 1
                            except:
                                pass
                        elif method == 'onehot':
                            try:
                                df_processed[column] = df_processed[column].fillna('MISSING')
                                dummies = pd.get_dummies(df_processed[column], prefix=column)
                                df_processed = pd.concat([df_processed.drop(column, axis=1), dummies], axis=1)
                                performance_metrics['encoding_applied'] += 1
                            except:
                                pass
                    
                    if valid_columns:
                        preprocessing_report.append({
                            'step': f'Applied {method} encoding to {len(valid_columns)} columns',
                            'details': f'Columns: {valid_columns[:5]}{"..." if len(valid_columns) > 5 else ""}',
                            'type': 'encoding',
                            'status': 'success'
                        })
                
                elif step_type == 'batch_dtype_conversion':
                    columns = step.get('columns', [])
                    new_dtype = step.get('dtype', 'numeric')
                    
                    valid_columns = [col for col in columns if col in df_processed.columns]
                    
                    for column in valid_columns:
                        try:
                            if new_dtype == 'numeric':
                                df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
                            elif new_dtype == 'datetime':
                                df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
                            elif new_dtype == 'category':
                                df_processed[column] = df_processed[column].astype('category')
                            elif new_dtype == 'string':
                                df_processed[column] = df_processed[column].astype('str')
                            performance_metrics['dtype_changes'] += 1
                        except:
                            pass
                    
                    if valid_columns:
                        preprocessing_report.append({
                            'step': f'Converted {len(valid_columns)} columns to {new_dtype}',
                            'details': f'Columns: {valid_columns[:5]}{"..." if len(valid_columns) > 5 else ""}',
                            'type': 'dtype_conversion',
                            'status': 'success'
                        })
                
                elif step_type == 'create_feature':
                    operation = step.get('operation')
                    column1 = step.get('column1')
                    column2 = step.get('column2')
                    new_column = step.get('new_column')
                    
                    if operation and column1 and column1 in df_processed.columns and new_column:
                        try:
                            if operation == 'add' and column2 and column2 in df_processed.columns:
                                df_processed[new_column] = df_processed[column1] + df_processed[column2]
                            elif operation == 'subtract' and column2 and column2 in df_processed.columns:
                                df_processed[new_column] = df_processed[column1] - df_processed[column2]
                            elif operation == 'multiply' and column2 and column2 in df_processed.columns:
                                df_processed[new_column] = df_processed[column1] * df_processed[column2]
                            elif operation == 'divide' and column2 and column2 in df_processed.columns:
                                df_processed[new_column] = df_processed[column1] / df_processed[column2].replace(0, np.nan)
                            elif operation == 'square':
                                df_processed[new_column] = df_processed[column1] ** 2
                            elif operation == 'sqrt':
                                df_processed[new_column] = np.sqrt(df_processed[column1].abs())
                            elif operation == 'log':
                                df_processed[new_column] = np.log(df_processed[column1].replace(0, np.nan).abs() + 1)
                            
                            preprocessing_report.append({
                                'step': f'Created new feature "{new_column}"',
                                'details': f'Operation: {operation} on {column1}' + (f' and {column2}' if column2 else ''),
                                'type': 'feature_engineering',
                                'status': 'success'
                            })
                        except Exception as e:
                            preprocessing_report.append({
                                'step': f'Failed to create feature "{new_column}"',
                                'details': str(e),
                                'type': 'feature_engineering',
                                'status': 'error'
                            })
            
            except Exception as e:
                preprocessing_report.append({
                    'step': f'Error processing step {step.get("type", "unknown")}',
                    'details': str(e),
                    'type': 'general',
                    'status': 'error'
                })
        
        # Save processed file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_filename = f'processed_{timestamp}_{os.path.basename(filename)}'
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        df_processed.to_csv(processed_path, index=False)
        
        # Calculate final statistics
        processed_info = {
            'rows': int(len(df_processed)),
            'columns': int(len(df_processed.columns)),
            'column_names': [str(col) for col in df_processed.columns.tolist()],
            'missing_values': int(df_processed.isnull().sum().sum()),
            'memory_usage_mb': float(df_processed.memory_usage(deep=True).sum() / 1024 / 1024),
            'original_shape': original_shape,
            'processed_shape': df_processed.shape,
            'preview': df_processed.head(10).replace({np.nan: None}).to_dict('records'),
            'performance_metrics': performance_metrics
        }
        
        # Convert preview data
        for record in processed_info['preview']:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    record[key] = float(value)
                elif pd.isna(value):
                    record[key] = None
        
        response_data = {
            'success': True,
            'processed_file': processed_filename,
            'processed_info': numpy_to_python(processed_info),
            'preprocessing_report': preprocessing_report,
            'message': f'Preprocessing completed. Dataset reduced from {original_shape[0]:,}×{original_shape[1]} to {len(df_processed):,}×{len(df_processed.columns)}.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in preprocess_dataset: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        original_file = data.get('original_file')
        processed_file = data.get('processed_file')
        target_column = data.get('target_column')
        selected_models = data.get('models', [])
        
        if not original_file or not processed_file or not target_column:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_file)
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_file)
        
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original file not found'}), 404
        if not os.path.exists(processed_path):
            return jsonify({'error': 'Processed file not found'}), 404
        
        df_original = pd.read_csv(original_path)
        df_processed = pd.read_csv(processed_path)
        
        if target_column not in df_original.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in original dataset'}), 400
        
        # Check if target column exists in processed data
        if target_column not in df_processed.columns:
            # Try to find alternative (might have been encoded)
            target_found = False
            for col in df_processed.columns:
                if target_column in col or col in target_column:
                    target_column = col
                    target_found = True
                    break
            
            if not target_found:
                return jsonify({'error': f'Target column "{target_column}" not found in processed dataset'}), 400
        
        def prepare_data(df, target_col):
            """Prepare data for model training"""
            if target_col not in df.columns:
                return None, None
            
            # Create a copy to avoid modifying original
            df_copy = df.copy()
            
            # Separate features and target
            X = df_copy.drop(columns=[target_col])
            y = df_copy[target_col]
            
            # Handle categorical columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                try:
                    # Simple label encoding for categoricals
                    X[col] = X[col].astype('category').cat.codes
                except:
                    # If encoding fails, drop the column
                    X = X.drop(columns=[col])
            
            # Handle missing values in features
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isnull().any():
                    X[col].fillna(X[col].median(), inplace=True)
            
            # Drop any remaining non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Handle missing values in target
            if y.isnull().any():
                # Remove rows with missing target
                valid_indices = y.notna()
                X = X[valid_indices]
                y = y[valid_indices]
            
            return X, y
        
        X_original, y_original = prepare_data(df_original, target_column)
        X_processed, y_processed = prepare_data(df_processed, target_column)
        
        if X_original is None or y_original is None or len(X_original) == 0:
            return jsonify({'error': 'Could not prepare original data for training'}), 400
        
        if X_processed is None or y_processed is None or len(X_processed) == 0:
            return jsonify({'error': 'Could not prepare processed data for training'}), 400
        
        # Check sample size
        min_samples = 10
        if len(X_original) < min_samples or len(X_processed) < min_samples:
            return jsonify({'error': f'Not enough data for training. Need at least {min_samples} samples.'}), 400
        
        problem_type = detect_problem_type(pd.concat([X_original, y_original], axis=1), target_column)
        
        # Adjust test size based on dataset size
        test_size = 0.2
        if len(X_original) < 100:
            test_size = 0.3
        elif len(X_original) < 50:
            test_size = 0.4
        
        # Split data
        try:
            if problem_type == 'classification':
                X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                    X_original, y_original, test_size=test_size, random_state=42, 
                    stratify=y_original
                )
                X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42,
                    stratify=y_processed
                )
            else:
                X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                    X_original, y_original, test_size=test_size, random_state=42
                )
                X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )
        except Exception as e:
            # If stratification fails, use regular split
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_original, y_original, test_size=test_size, random_state=42
            )
            X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=42
            )
        
        results = []
        
        for model_name in selected_models:
            model_result = {
                'model': model_name,
                'original_accuracy': None,
                'original_accuracy_percent': None,
                'processed_accuracy': None,
                'processed_accuracy_percent': None,
                'improvement': None,
                'improvement_percent': None,
                'additional_metrics': {},
                'error': None
            }
            
            try:
                # Select appropriate model based on problem type
                if problem_type == 'classification':
                    if model_name == 'logistic_regression':
                        model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
                    elif model_name == 'random_forest':
                        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
                    elif model_name == 'decision_tree':
                        model = DecisionTreeClassifier(random_state=42, max_depth=5)
                    elif model_name == 'svm':
                        model = SVC(kernel='rbf', probability=True, random_state=42)
                    elif model_name == 'kmeans':
                        # KMeans for clustering (unsupervised)
                        n_clusters = min(10, len(np.unique(y_original)))
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        
                        # Train on original data
                        model.fit(X_train_orig)
                        labels_orig = model.predict(X_test_orig)
                        if len(np.unique(labels_orig)) > 1:
                            score_orig = silhouette_score(X_test_orig, labels_orig)
                            model_result['original_accuracy'] = float(score_orig)
                            model_result['original_accuracy_percent'] = float(score_orig * 100)
                        
                        # Train on processed data
                        model.fit(X_train_proc)
                        labels_proc = model.predict(X_test_proc)
                        if len(np.unique(labels_proc)) > 1:
                            score_proc = silhouette_score(X_test_proc, labels_proc)
                            model_result['processed_accuracy'] = float(score_proc)
                            model_result['processed_accuracy_percent'] = float(score_proc * 100)
                        
                        if model_result['original_accuracy'] is not None and model_result['processed_accuracy'] is not None:
                            improvement = float(model_result['processed_accuracy'] - model_result['original_accuracy'])
                            model_result['improvement'] = improvement
                            model_result['improvement_percent'] = float(improvement * 100)
                        
                        results.append(model_result)
                        continue
                    elif model_name == 'linear_regression':
                        model_result['error'] = 'Linear Regression not suitable for classification problems'
                        results.append(model_result)
                        continue
                
                else:  # regression
                    if model_name == 'linear_regression':
                        model = LinearRegression()
                    elif model_name == 'random_forest':
                        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    elif model_name == 'decision_tree':
                        model = DecisionTreeRegressor(random_state=42, max_depth=5)
                    elif model_name == 'svm':
                        model = SVR(kernel='rbf')
                    elif model_name == 'kmeans':
                        model_result['error'] = 'K-Means not suitable for regression problems'
                        results.append(model_result)
                        continue
                    elif model_name == 'logistic_regression':
                        model_result['error'] = 'Logistic Regression not suitable for regression problems'
                        results.append(model_result)
                        continue
                
                # Train and evaluate on original data
                model.fit(X_train_orig, y_train_orig)
                y_pred_orig = model.predict(X_test_orig)
                
                if problem_type == 'classification':
                    acc_orig = accuracy_score(y_test_orig, y_pred_orig)
                    try:
                        model_result['additional_metrics']['original_precision'] = float(precision_score(
                            y_test_orig, y_pred_orig, average='weighted', zero_division=0))
                        model_result['additional_metrics']['original_recall'] = float(recall_score(
                            y_test_orig, y_pred_orig, average='weighted', zero_division=0))
                        model_result['additional_metrics']['original_f1'] = float(f1_score(
                            y_test_orig, y_pred_orig, average='weighted', zero_division=0))
                    except:
                        pass
                else:
                    acc_orig = r2_score(y_test_orig, y_pred_orig)
                    try:
                        model_result['additional_metrics']['original_mse'] = float(mean_squared_error(y_test_orig, y_pred_orig))
                        model_result['additional_metrics']['original_rmse'] = float(math.sqrt(mean_squared_error(y_test_orig, y_pred_orig)))
                        model_result['additional_metrics']['original_mae'] = float(np.mean(np.abs(y_test_orig - y_pred_orig)))
                    except:
                        pass
                
                model_result['original_accuracy'] = float(acc_orig)
                model_result['original_accuracy_percent'] = float(acc_orig * 100) if problem_type == 'classification' else float(acc_orig * 100)
                
                # Train and evaluate on processed data
                model.fit(X_train_proc, y_train_proc)
                y_pred_proc = model.predict(X_test_proc)
                
                if problem_type == 'classification':
                    acc_proc = accuracy_score(y_test_proc, y_pred_proc)
                    try:
                        model_result['additional_metrics']['processed_precision'] = float(precision_score(
                            y_test_proc, y_pred_proc, average='weighted', zero_division=0))
                        model_result['additional_metrics']['processed_recall'] = float(recall_score(
                            y_test_proc, y_pred_proc, average='weighted', zero_division=0))
                        model_result['additional_metrics']['processed_f1'] = float(f1_score(
                            y_test_proc, y_pred_proc, average='weighted', zero_division=0))
                    except:
                        pass
                else:
                    acc_proc = r2_score(y_test_proc, y_pred_proc)
                    try:
                        model_result['additional_metrics']['processed_mse'] = float(mean_squared_error(y_test_proc, y_pred_proc))
                        model_result['additional_metrics']['processed_rmse'] = float(math.sqrt(mean_squared_error(y_test_proc, y_pred_proc)))
                        model_result['additional_metrics']['processed_mae'] = float(np.mean(np.abs(y_test_proc - y_pred_proc)))
                    except:
                        pass
                
                model_result['processed_accuracy'] = float(acc_proc)
                model_result['processed_accuracy_percent'] = float(acc_proc * 100) if problem_type == 'classification' else float(acc_proc * 100)
                
                # Calculate improvement
                if model_result['original_accuracy'] is not None and model_result['processed_accuracy'] is not None:
                    improvement = float(model_result['processed_accuracy'] - model_result['original_accuracy'])
                    model_result['improvement'] = improvement
                    model_result['improvement_percent'] = float(improvement * 100)
                
            except Exception as e:
                model_result['error'] = str(e)
                print(f"Error training {model_name}: {e}")
                traceback.print_exc()
            
            results.append(model_result)
        
        # Calculate summary statistics
        valid_results = [r for r in results if r['improvement'] is not None]
        if valid_results:
            avg_improvement = np.mean([r['improvement'] for r in valid_results])
            avg_improvement_percent = np.mean([r['improvement_percent'] for r in valid_results])
            best_model = max(valid_results, key=lambda x: x['processed_accuracy'] if x['processed_accuracy'] is not None else -1)
            worst_model = min(valid_results, key=lambda x: x['processed_accuracy'] if x['processed_accuracy'] is not None else 1)
        else:
            avg_improvement = 0
            avg_improvement_percent = 0
            best_model = None
            worst_model = None
        
        response_data = {
            'success': True,
            'problem_type': problem_type,
            'results': numpy_to_python(results),
            'original_shape': [int(dim) for dim in df_original.shape],
            'processed_shape': [int(dim) for dim in df_processed.shape],
            'summary': {
                'average_improvement': float(avg_improvement),
                'average_improvement_percent': float(avg_improvement_percent),
                'best_model': best_model['model'] if best_model else None,
                'best_accuracy': float(best_model['processed_accuracy']) if best_model else None,
                'best_accuracy_percent': float(best_model['processed_accuracy_percent']) if best_model else None,
                'worst_model': worst_model['model'] if worst_model else None,
                'worst_accuracy': float(worst_model['processed_accuracy']) if worst_model else None,
                'models_trained': len(valid_results),
                'models_failed': len(results) - len(valid_results)
            },
            'message': f'Model training completed. Problem type: {problem_type}. {len(valid_results)} models trained successfully.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        # Check in both directories
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        
        if os.path.exists(filepath):
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype='text/csv'
            )
        
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    try:
        # Clean up old files (older than 24 hours)
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    file_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                    if file_time < cutoff_time:
                        os.remove(filepath)
        
        return jsonify({
            'success': True, 
            'message': 'Old files cleaned up (older than 24 hours)'
        })
    except Exception as e:
        print(f"Error cleaning up files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'service': 'ML Preprocessing App',
        'version': '2.0'
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
