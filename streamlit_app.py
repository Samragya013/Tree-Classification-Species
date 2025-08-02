from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from collections import Counter
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os
import sqlite3
from datetime import datetime
import logging
from skimage import exposure
from functools import lru_cache
import validators

app = Flask(__name__, template_folder='/content/templates', static_folder='/content/static')
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')  # Use environment variable for security
app.config['UPLOAD_FOLDER'] = '/content/static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SQLITE_DB'] = '/content/users.db'

# Setup logging
logging.basicConfig(filename='/content/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database operations
def init_db():
    """Initialize SQLite database for users and predictions."""
    try:
        with sqlite3.connect(app.config['SQLITE_DB']) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         username TEXT UNIQUE NOT NULL,
                         password TEXT NOT NULL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS predictions
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER,
                         prediction_type TEXT,
                         input_data TEXT,
                         result TEXT,
                         timestamp DATETIME,
                         FOREIGN KEY (user_id) REFERENCES users(id))''')
            conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Database initialization error: {str(e)}")
        raise

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Cached data and model loading
@lru_cache(maxsize=1)
def load_data():
    """Load and cache tree dataset."""
    try:
        return pd.read_pickle('/content/tree_data.pkl')
    except FileNotFoundError:
        logging.error("tree_data.pkl not found")
        raise FileNotFoundError("Tree dataset file is missing")

@lru_cache(maxsize=1)
def load_nn_models():
    """Load and cache nearest neighbors model and scaler."""
    try:
        scaler = joblib.load('/content/scaler.joblib')
        nn_model = joblib.load('/content/nn_model.joblib')
        return scaler, nn_model
    except FileNotFoundError:
        logging.error("Scaler or NN model file not found")
        raise FileNotFoundError("Model files are missing")

@lru_cache(maxsize=1)
def load_cnn_model():
    """Load and cache CNN model."""
    try:
        return load_model("/content/basic_cnn_tree_species.h5")
    except FileNotFoundError:
        logging.error("CNN model file not found")
        raise FileNotFoundError("CNN model file is missing")

# Utility functions
def recommend_species(input_data, nn_model, scaler, df, top_n=5):
    """Generate tree species recommendations based on input features."""
    try:
        input_scaled = scaler.transform([input_data])
        distances, indices = nn_model.kneighbors(input_scaled, n_neighbors=top_n)
        neighbors = df.iloc[indices[0]]
        species_counts = Counter(neighbors['common_name'])
        return species_counts.most_common(top_n)
    except Exception as e:
        logging.error(f"Recommendation error: {str(e)}")
        raise ValueError("Failed to generate recommendations")

def get_common_locations_for_species(df, tree_name, top_n=10):
    """Retrieve common locations for a given tree species."""
    try:
        species_df = df[df['common_name'] == tree_name]
        if species_df.empty:
            return pd.DataFrame(columns=['city', 'state', 'count', 'latitude', 'longitude'])
        return species_df.groupby(['city', 'state', 'latitude', 'longitude']) \
                        .size().reset_index(name='count') \
                        .sort_values(by='count', ascending=False) \
                        .head(top_n)
    except Exception as e:
        logging.error(f"Location retrieval error: {str(e)}")
        raise ValueError("Failed to retrieve locations")

def preprocess_image(image, enhance_contrast=False, denoise=False):
    """Preprocess image for CNN prediction."""
    try:
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        if enhance_contrast:
            img_array = exposure.equalize_hist(img_array)
        if denoise:
            img_array = exposure.rescale_intensity(img_array)
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        raise ValueError("Failed to preprocess image")

def validate_input(data, expected_types):
    """Validate input data types and ranges."""
    for key, value in data.items():
        expected_type, range_min, range_max = expected_types.get(key, (None, None, None))
        try:
            value = expected_type(value)
            if range_min is not None and range_max is not None:
                if not (range_min <= value <= range_max):
                    raise ValueError(f"{key} out of range [{range_min}, {range_max}]")
            return value
        except (ValueError, TypeError):
            raise ValueError(f"Invalid {key}: {value}")

# Routes
@login_manager.user_loader
def load_user(user_id):
    try:
        with sqlite3.connect(app.config['SQLITE_DB']) as conn:
            c = conn.cursor()
            c.execute('SELECT id, username FROM users WHERE id = ?', (user_id,))
            user_data = c.fetchone()
        if user_data:
            return User(user_data[0], user_data[1])
        return None
    except sqlite3.Error as e:
        logging.error(f"User load error: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('register.html')
        if len(username) < 3 or len(password) < 6:
            flash('Username must be at least 3 characters and password at least 6 characters.', 'danger')
            return render_template('register.html')
        try:
            with sqlite3.connect(app.config['SQLITE_DB']) as conn:
                c = conn.cursor()
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)',
                         (username, generate_password_hash(password)))
                conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'danger')
        except sqlite3.Error as e:
            logging.error(f"Registration database error: {str(e)}")
            flash('An error occurred during registration.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('login.html')
        try:
            with sqlite3.connect(app.config['SQLITE_DB']) as conn:
                c = conn.cursor()
                c.execute('SELECT id, username, password FROM users WHERE username = ?', (username,))
                user_data = c.fetchone()
            if user_data and check_password_hash(user_data[2], password):
                user = User(user_data[0], user_data[1])
                login_user(user)
                flash('Logged in successfully!', 'success')
                return redirect(url_for('index'))
            flash('Invalid username or password.', 'danger')
        except sqlite3.Error as e:
            logging.error(f"Login database error: {str(e)}")
            flash('An error occurred during login.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    try:
        df = load_data()
        scaler, nn_model = load_nn_models()
        class_labels = sorted(df['common_name'].unique())
        cities = sorted(df['city'].unique())
        states = sorted(df['state'].unique())
        natives = sorted(df['native'].unique())

        if request.method == 'POST':
            # Input validation
            input_data = {
                'latitude': request.form.get('latitude'),
                'longitude': request.form.get('longitude'),
                'diameter': request.form.get('diameter'),
                'native': request.form.get('native'),
                'city': request.form.get('city'),
                'state': request.form.get('state')
            }
            expected_types = {
                'latitude': (float, -90.0, 90.0),
                'longitude': (float, -180.0, 180.0),
                'diameter': (float, 0.0, 1000.0),
                'native': (str, None, None),
                'city': (str, None, None),
                'state': (str, None, None)
            }
            try:
                validated_data = {k: validate_input({k: v}, expected_types) for k, v in input_data.items()}
                if validated_data['native'] not in natives or validated_data['city'] not in cities or validated_data['state'] not in states:
                    raise ValueError("Invalid category selection")
                
                native_code = df['native'].astype('category').cat.categories.get_loc(validated_data['native'])
                city_code = df['city'].astype('category').cat.categories.get_loc(validated_data['city'])
                state_code = df['state'].astype('category').cat.categories.get_loc(validated_data['state'])

                input_array = [
                    validated_data['latitude'],
                    validated_data['longitude'],
                    validated_data['diameter'],
                    native_code,
                    city_code,
                    state_code
                ]
                recommendations = recommend_species(input_array, nn_model, scaler, df)

                # Save to prediction history
                with sqlite3.connect(app.config['SQLITE_DB']) as conn:
                    c = conn.cursor()
                    c.execute('INSERT INTO predictions (user_id, prediction_type, input_data, result, timestamp) VALUES (?, ?, ?, ?, ?)',
                             (current_user.id, 'recommend', str(input_array), str(recommendations), datetime.now()))
                    conn.commit()

                return render_template('recommend.html', recommendations=recommendations, cities=cities, states=states, natives=natives)
            except ValueError as e:
                flash(str(e), 'danger')
            except Exception as e:
                logging.error(f"Recommendation processing error: {str(e)}")
                flash('An error occurred while processing your request.', 'danger')

        return render_template('recommend.html', cities=cities, states=states, natives=natives)
    except Exception as e:
        logging.error(f"Recommendation route error: {str(e)}")
        flash('Failed to load recommendation system.', 'danger')
        return render_template('recommend.html', cities=[], states=[], natives=[])

@app.route('/locations', methods=['GET', 'POST'])
@login_required
def locations():
    try:
        df = load_data()
        class_labels = sorted(df['common_name'].unique())

        if request.method == 'POST':
            tree_name = request.form.get('tree_name')
            if not tree_name or tree_name not in class_labels:
                flash('Please select a valid tree species.', 'danger')
                return render_template('locations.html', species=class_labels)

            locations = get_common_locations_for_species(df, tree_name)
            
            # Save to prediction history
            with sqlite3.connect(app.config['SQLITE_DB']) as conn:
                c = conn.cursor()
                c.execute('INSERT INTO predictions (user_id, prediction_type, input_data, result, timestamp) VALUES (?, ?, ?, ?, ?)',
                         (current_user.id, 'locations', tree_name, locations.to_json(), datetime.now()))
                conn.commit()

            map_data = locations.to_dict('records') if not locations.empty else []
            
            if request.form.get('export'):
                output = io.StringIO()
                locations.to_csv(output, index=False)
                output.seek(0)
                return send_file(
                    io.BytesIO(output.getvalue().encode('utf-8')),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f'{tree_name}_locations.csv'
                )

            return render_template('locations.html', tree_name=tree_name, locations=locations, map_data=map_data, species=class_labels)
        
        return render_template('locations.html', species=class_labels)
    except Exception as e:
        logging.error(f"Locations route error: {str(e)}")
        flash('Failed to load location data.', 'danger')
        return render_template('locations.html', species=[])

@app.route('/identify', methods=['GET', 'POST'])
@login_required
def identify():
    try:
        df = load_data()
        cnn_model = load_cnn_model()
        class_labels = sorted(df['common_name'].unique())

        if request.method == 'POST':
            if 'image' not in request.files:
                flash('No image file uploaded.', 'danger')
                return render_template('identify.html', species=class_labels)

            file = request.files['image']
            if file.filename == '':
                flash('No file selected.', 'danger')
                return render_template('identify.html', species=class_labels)

            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                flash('Only JPG and PNG files are allowed.', 'danger')
                return render_template('identify.html', species=class_labels)

            try:
                enhance_contrast = 'enhance_contrast' in request.form
                denoise = 'denoise' in request.form
                image = Image.open(file).convert('RGB')
                img_array = preprocess_image(image, enhance_contrast, denoise)
                
                predictions = cnn_model.predict(img_array, verbose=0)
                pred_idx = np.argmax(predictions)
                pred_label = class_labels[pred_idx]
                confidence = predictions[0][pred_idx]
                top_3_idx = predictions[0].argsort()[-3:][::-1]
                top_3 = [(class_labels[i], predictions[0][i]) for i in top_3_idx]
                
                locations = get_common_locations_for_species(df, pred_label)
                
                # Save to prediction history
                with sqlite3.connect(app.config['SQLITE_DB']) as conn:
                    c = conn.cursor()
                    c.execute('INSERT INTO predictions (user_id, prediction_type, input_data, result, timestamp) VALUES (?, ?, ?, ?, ?)',
                             (current_user.id, 'identify', 'image_upload', f"{pred_label}:{confidence:.2f}", datetime.now()))
                    conn.commit()

                # Save uploaded image
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{current_user.id}.jpg')
                image.save(image_path)

                return render_template('identify.html',
                                     prediction=pred_label,
                                     confidence=confidence,
                                     top_3=top_3,
                                     locations=locations,
                                     map_data=locations.to_dict('records') if not locations.empty else [],
                                     image_path=f'uploads/temp_{current_user.id}.jpg',
                                     species=class_labels)
            except Exception as e:
                logging.error(f"Image identification error: {str(e)}")
                flash('An error occurred while processing the image.', 'danger')

        return render_template('identify.html', species=class_labels)
    except Exception as e:
        logging.error(f"Identify route error: {str(e)}")
        flash('Failed to load identification system.', 'danger')
        return render_template('identify.html', species=[])

@app.route('/history')
@login_required
def history():
    try:
        with sqlite3.connect(app.config['SQLITE_DB']) as conn:
            c = conn.cursor()
            c.execute('SELECT prediction_type, input_data, result, timestamp FROM predictions WHERE user_id = ? ORDER BY timestamp DESC',
                     (current_user.id,))
            predictions = c.fetchall()
        return render_template('history.html', predictions=predictions)
    except sqlite3.Error as e:
        logging.error(f"History retrieval error: {str(e)}")
        flash('Failed to load prediction history.', 'danger')
        return render_template('history.html', predictions=[])

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)  
