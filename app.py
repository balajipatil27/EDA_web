import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, precision_score, recall_score, f1_score
import traceback
import warnings
import datetime
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'ml-preprocessor-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# Create uploads directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    
    unique_values = df[target_column].nunique()
    
    if unique_values <= 10 or df[target_column].dtype == 'object':
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
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            try:
                pd.to_numeric(df[col])
                numerical_cols.append(col)
            except:
                categorical_cols.append(col)
    
    return {
        'categorical': categorical_cols,
        'numerical': numerical_cols
    }

def load_dataset(filepath):
    """Load dataset from various formats"""
    try:
        if filepath.endswith('.csv'):
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding)
                    for col in df.select_dtypes(include=['object']).columns:
                        try:
                            df[col] = pd.to_datetime(df[col], errors='ignore')
                        except:
                            pass
                    return df
                except:
                    continue
            return pd.read_csv(filepath)
        elif filepath.endswith(('.xlsx', '.xls')):
            try:
                return pd.read_excel(filepath, engine='openpyxl')
            except:
                try:
                    return pd.read_excel(filepath, engine='xlrd')
                except:
                    return None
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

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
                return jsonify({'error': 'Failed to read the file or file is empty.'}), 400
            
            print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
            
            column_types = detect_column_types(df)
            
            for col in df.columns:
                if pd.api.types.is_integer_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='integer')
                elif pd.api.types.is_float_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], downcast='float')
            
            missing_values_sum = df.isnull().sum()
            missing_percentage = (missing_values_sum / len(df) * 100)
            
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
                'preview': df.head(10).replace({np.nan: None}).to_dict('records')
            }
            
            for record in dataset_info['preview']:
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.int64)):
                        record[key] = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        record[key] = float(value)
                    elif pd.isna(value):
                        record[key] = None
            
            original_filename = 'original_' + filename
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
            print(traceback.format_exc())
            
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
            'data_quality_metrics': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            basic_stats = df[numeric_cols].describe()
            analysis_report['basic_stats'] = {
                str(col): {
                    str(stat): float(val) if not pd.isna(val) else None 
                    for stat, val in stats.items()
                }
                for col, stats in basic_stats.to_dict().items()
            }
        
        for column in df.columns:
            col_analysis = {
                'name': str(column),
                'dtype': str(df[column].dtype),
                'unique_values': int(df[column].nunique()),
                'missing_count': int(df[column].isnull().sum()),
                'missing_percentage': float((df[column].isnull().sum() / len(df)) * 100)
            }
            
            if df[column].dtype in ['object', 'category']:
                value_counts = df[column].value_counts().head(5)
                col_analysis['top_values'] = {str(k): int(v) for k, v in value_counts.items()}
                col_analysis['type'] = 'categorical'
            elif pd.api.types.is_numeric_dtype(df[column]):
                col_analysis['min'] = float(df[column].min()) if not pd.isna(df[column].min()) else None
                col_analysis['max'] = float(df[column].max()) if not pd.isna(df[column].max()) else None
                col_analysis['mean'] = float(df[column].mean()) if not pd.isna(df[column].mean()) else None
                col_analysis['median'] = float(df[column].median()) if not pd.isna(df[column].median()) else None
                col_analysis['std'] = float(df[column].std()) if not pd.isna(df[column].std()) else None
                col_analysis['type'] = 'numerical'
            else:
                col_analysis['type'] = 'other'
            
            analysis_report['column_analysis'].append(col_analysis)
        
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
            'columns_high_missing': int(((df.isnull().sum() / len(df)) > 0.5).sum())
        }
        
        if len(numeric_cols) > 1:
            try:
                correlation = df[numeric_cols].corr()
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
                                        'correlation': float(corr_value)
                                    })
                    analysis_report['correlation_matrix'] = corr_records
            except Exception as e:
                print(f"Error calculating correlation: {e}")
        
        return jsonify({
            'success': True,
            'analysis': numpy_to_python(analysis_report)
        })
        
    except Exception as e:
        print(f"Error in analyze_dataset: {str(e)}")
        print(traceback.format_exc())
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
            'outlier_suggestions': []
        }
        
        for col in column_types['categorical']:
            unique_count = df[col].nunique()
            if 2 <= unique_count <= 20:
                suggestions['encoding_suggestions'].append({
                    'column': col,
                    'unique_values': int(unique_count),
                    'suggested_method': 'label' if unique_count <= 10 else 'onehot'
                })
        
        for col in column_types['categorical']:
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.8:
                    suggestions['dtype_conversion_suggestions'].append({
                        'column': col,
                        'current_dtype': str(df[col].dtype),
                        'suggested_dtype': 'numeric'
                    })
            except:
                pass
        
        for col in column_types['numerical']:
            if df[col].notna().sum() > 0:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outlier_count > 0:
                        suggestions['outlier_suggestions'].append({
                            'column': col,
                            'outlier_count': int(outlier_count),
                            'outlier_percentage': float((outlier_count / len(df)) * 100)
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
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original file not found'}), 404
        
        df = pd.read_csv(original_path)
        
        df_processed = df.copy()
        preprocessing_report = []
        performance_metrics = {
            'rows_removed': 0,
            'columns_removed': 0,
            'missing_values_filled': 0,
            'outliers_removed': 0,
            'encoding_applied': 0,
            'dtype_changes': 0
        }
        
        for step in preprocessing_steps:
            try:
                if step['type'] == 'drop_high_missing':
                    threshold = step.get('threshold', 50)
                    missing_percentage = (df_processed.isnull().sum() / len(df_processed)) * 100
                    columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
                    if columns_to_drop:
                        df_processed.drop(columns=columns_to_drop, inplace=True)
                        performance_metrics['columns_removed'] += len(columns_to_drop)
                        preprocessing_report.append({
                            'step': f'Dropped columns with >{threshold}% missing values',
                            'details': f'Columns removed: {columns_to_drop}'
                        })
                
                elif step['type'] == 'remove_duplicates':
                    before = len(df_processed)
                    df_processed.drop_duplicates(inplace=True)
                    after = len(df_processed)
                    removed = before - after
                    if removed > 0:
                        performance_metrics['rows_removed'] += removed
                        preprocessing_report.append({
                            'step': 'Removed duplicate rows',
                            'details': f'Removed {removed} duplicate rows'
                        })
                
                elif step['type'] == 'change_dtype':
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
                            
                            performance_metrics['dtype_changes'] += 1
                            preprocessing_report.append({
                                'step': f'Changed data type of column "{column}"',
                                'details': f'From {old_dtype} to {new_dtype}'
                            })
                        except Exception as e:
                            preprocessing_report.append({
                                'step': f'Failed to change data type of column "{column}"',
                                'details': str(e)
                            })
                
                elif step['type'] == 'encoding':
                    column = step.get('column')
                    method = step.get('method')
                    
                    if column and column in df_processed.columns and method:
                        if method == 'label':
                            try:
                                le = LabelEncoder()
                                df_processed[column] = le.fit_transform(df_processed[column].astype(str))
                                performance_metrics['encoding_applied'] += 1
                                preprocessing_report.append({
                                    'step': f'Applied Label Encoding to column "{column}"',
                                    'details': f'Number of unique values: {len(le.classes_)}'
                                })
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to apply Label Encoding to column "{column}"',
                                    'details': str(e)
                                })
                        elif method == 'onehot':
                            try:
                                dummies = pd.get_dummies(df_processed[column], prefix=column, drop_first=True)
                                df_processed = pd.concat([df_processed.drop(column, axis=1), dummies], axis=1)
                                performance_metrics['encoding_applied'] += 1
                                preprocessing_report.append({
                                    'step': f'Applied One-Hot Encoding to column "{column}"',
                                    'details': f'Created {len(dummies.columns)} new columns'
                                })
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to apply One-Hot Encoding to column "{column}"',
                                    'details': str(e)
                                })
                
                elif step['type'] == 'handle_missing':
                    column = step.get('column')
                    method = step.get('method')
                    
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
                                'details': f'Rows dropped: {removed}, Rows remaining: {after}'
                            })
                        else:
                            if method == 'mean':
                                fill_value = df_processed[column].mean()
                            elif method == 'median':
                                fill_value = df_processed[column].median()
                            elif method == 'mode':
                                fill_value = df_processed[column].mode()[0] if not df_processed[column].mode().empty else 0
                            else:
                                fill_value = step.get('custom_value', 0)
                            
                            df_processed[column].fillna(fill_value, inplace=True)
                            performance_metrics['missing_values_filled'] += int(missing_count)
                            preprocessing_report.append({
                                'step': f'Filled missing values in column "{column}"',
                                'details': f'Method: {method}, Fill value: {fill_value}, Rows filled: {int(missing_count)}'
                            })
                
                elif step['type'] == 'remove_outliers':
                    column = step.get('column')
                    
                    if column and column in df_processed.columns:
                        if pd.api.types.is_numeric_dtype(df_processed[column]):
                            try:
                                Q1 = df_processed[column].quantile(0.25)
                                Q3 = df_processed[column].quantile(0.75)
                                IQR = Q3 - Q1
                                
                                if IQR > 0:
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                    
                                    before = len(df_processed)
                                    df_processed = df_processed[
                                        (df_processed[column] >= lower_bound) & 
                                        (df_processed[column] <= upper_bound)
                                    ]
                                    after = len(df_processed)
                                    removed = before - after
                                    
                                    if removed > 0:
                                        performance_metrics['outliers_removed'] += removed
                                        preprocessing_report.append({
                                            'step': f'Removed outliers from column "{column}"',
                                            'details': f'Removed {removed} rows using IQR method'
                                        })
                            except Exception as e:
                                preprocessing_report.append({
                                    'step': f'Failed to remove outliers from column "{column}"',
                                    'details': str(e)
                                })
                
                elif step['type'] == 'batch_encoding':
                    columns = step.get('columns', [])
                    method = step.get('method', 'label')
                    
                    for column in columns:
                        if column in df_processed.columns:
                            if method == 'label':
                                try:
                                    le = LabelEncoder()
                                    df_processed[column] = le.fit_transform(df_processed[column].astype(str))
                                    performance_metrics['encoding_applied'] += 1
                                except:
                                    pass
                            elif method == 'onehot':
                                try:
                                    dummies = pd.get_dummies(df_processed[column], prefix=column, drop_first=True)
                                    df_processed = pd.concat([df_processed.drop(column, axis=1), dummies], axis=1)
                                    performance_metrics['encoding_applied'] += 1
                                except:
                                    pass
                    
                    if columns:
                        preprocessing_report.append({
                            'step': f'Applied {method} encoding to {len(columns)} columns',
                            'details': f'Columns: {columns}'
                        })
                
                elif step['type'] == 'batch_dtype_conversion':
                    columns = step.get('columns', [])
                    new_dtype = step.get('dtype', 'numeric')
                    
                    for column in columns:
                        if column in df_processed.columns:
                            try:
                                if new_dtype == 'numeric':
                                    df_processed[column] = pd.to_numeric(df_processed[column], errors='coerce')
                                elif new_dtype == 'datetime':
                                    df_processed[column] = pd.to_datetime(df_processed[column], errors='coerce')
                                elif new_dtype == 'category':
                                    df_processed[column] = df_processed[column].astype('category')
                                performance_metrics['dtype_changes'] += 1
                            except:
                                pass
                    
                    if columns:
                        preprocessing_report.append({
                            'step': f'Converted {len(columns)} columns to {new_dtype}',
                            'details': f'Columns: {columns}'
                        })
                
            except Exception as e:
                preprocessing_report.append({
                    'step': f'Error processing step {step.get("type", "unknown")}',
                    'details': str(e)
                })
        
        processed_filename = 'processed_' + filename
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        df_processed.to_csv(processed_path, index=False)
        
        processed_info = {
            'rows': int(len(df_processed)),
            'columns': int(len(df_processed.columns)),
            'column_names': [str(col) for col in df_processed.columns.tolist()],
            'missing_values': int(df_processed.isnull().sum().sum()),
            'preview': df_processed.head(10).replace({np.nan: None}).to_dict('records'),
            'performance_metrics': performance_metrics
        }
        
        for record in processed_info['preview']:
            for key, value in record.items():
                if isinstance(value, (np.integer, np.int64)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    record[key] = float(value)
        
        response_data = {
            'success': True,
            'processed_file': processed_filename,
            'processed_info': numpy_to_python(processed_info),
            'preprocessing_report': preprocessing_report,
            'message': f'Preprocessing completed. Processed dataset has {len(df_processed):,} rows and {len(df_processed.columns)} columns.'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in preprocess_dataset: {str(e)}")
        print(traceback.format_exc())
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
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_file)
        
        if not os.path.exists(original_path):
            return jsonify({'error': 'Original file not found'}), 404
        if not os.path.exists(processed_path):
            return jsonify({'error': 'Processed file not found'}), 404
        
        df_original = pd.read_csv(original_path)
        df_processed = pd.read_csv(processed_path)
        
        if target_column not in df_original.columns:
            return jsonify({'error': f'Target column "{target_column}" not found in dataset'}), 400
        
        def prepare_data(df, target_col):
            df = df.dropna(subset=[target_col])
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                try:
                    X[col] = X[col].astype('category').cat.codes
                except:
                    X = X.drop(columns=[col])
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X[col].fillna(X[col].median(), inplace=True)
            
            return X, y
        
        X_original, y_original = prepare_data(df_original, target_column)
        X_processed, y_processed = prepare_data(df_processed, target_column)
        
        if len(X_original) < 10 or len(X_processed) < 10:
            return jsonify({'error': 'Not enough data for training. Need at least 10 samples.'}), 400
        
        problem_type = detect_problem_type(df_original, target_column)
        
        test_size = 0.2
        if len(X_original) < 100:
            test_size = 0.3
        
        try:
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_original, y_original, test_size=test_size, random_state=42, 
                stratify=y_original if problem_type == 'classification' else None
            )
        except:
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_original, y_original, test_size=test_size, random_state=42
            )
        
        try:
            X_train_proc, X_test_proc, y_train_proc, y_test_proc = train_test_split(
                X_processed, y_processed, test_size=test_size, random_state=42,
                stratify=y_processed if problem_type == 'classification' else None
            )
        except:
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
                'additional_metrics': {}
            }
            
            try:
                if problem_type == 'classification':
                    if model_name == 'logistic_regression':
                        model = LogisticRegression(max_iter=1000, random_state=42)
                    elif model_name == 'random_forest':
                        model = RandomForestClassifier(n_estimators=50, random_state=42)
                    elif model_name == 'decision_tree':
                        model = DecisionTreeClassifier(random_state=42, max_depth=5)
                    elif model_name == 'svm':
                        model = SVC(kernel='linear', probability=True, random_state=42)
                    elif model_name == 'kmeans':
                        n_clusters = min(10, len(np.unique(y_original)))
                        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        
                        model.fit(X_train_orig)
                        labels_orig = model.predict(X_test_orig)
                        if len(np.unique(labels_orig)) > 1:
                            score_orig = silhouette_score(X_test_orig, labels_orig)
                            model_result['original_accuracy'] = float(score_orig)
                            model_result['original_accuracy_percent'] = float(score_orig * 100)
                        
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
                        model_result['error'] = 'Linear Regression not suitable for classification'
                        results.append(model_result)
                        continue
                
                else:
                    if model_name == 'linear_regression':
                        model = LinearRegression()
                    elif model_name == 'random_forest':
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                    elif model_name == 'decision_tree':
                        model = DecisionTreeRegressor(random_state=42, max_depth=5)
                    elif model_name == 'svm':
                        model = SVR(kernel='linear')
                    elif model_name == 'kmeans':
                        model_result['error'] = 'K-Means not suitable for regression'
                        results.append(model_result)
                        continue
                    elif model_name == 'logistic_regression':
                        model_result['error'] = 'Logistic Regression not suitable for regression'
                        results.append(model_result)
                        continue
                
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
                
                model_result['original_accuracy'] = float(acc_orig)
                model_result['original_accuracy_percent'] = float(acc_orig * 100) if problem_type == 'classification' else float(acc_orig * 100)
                
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
                
                model_result['processed_accuracy'] = float(acc_proc)
                model_result['processed_accuracy_percent'] = float(acc_proc * 100) if problem_type == 'classification' else float(acc_proc * 100)
                
                if model_result['original_accuracy'] is not None and model_result['processed_accuracy'] is not None:
                    improvement = float(model_result['processed_accuracy'] - model_result['original_accuracy'])
                    model_result['improvement'] = improvement
                    model_result['improvement_percent'] = float(improvement * 100)
                
            except Exception as e:
                model_result['error'] = str(e)
                print(f"Error training {model_name}: {e}")
            
            results.append(model_result)
        
        valid_results = [r for r in results if r['improvement'] is not None]
        if valid_results:
            avg_improvement = np.mean([r['improvement'] for r in valid_results])
            avg_improvement_percent = np.mean([r['improvement_percent'] for r in valid_results])
            best_model = max(valid_results, key=lambda x: x['processed_accuracy'] if x['processed_accuracy'] is not None else -1)
        else:
            avg_improvement = 0
            avg_improvement_percent = 0
            best_model = None
        
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
                'best_accuracy_percent': float(best_model['processed_accuracy_percent']) if best_model else None
            },
            'message': f'Model training completed. Problem type: {problem_type}'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
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
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        return jsonify({'success': True, 'message': 'All files cleaned up'})
    except Exception as e:
        print(f"Error cleaning up files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'service': 'ML Preprocessing App'
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