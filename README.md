# Context-Aware Emoji Prediction from Text Using NLP Techniques

This is a complete application that predicts emojis from text using natural language processing (NLP). It works in real-time and can also be used as a Chrome extension.

## What This Project Does

Imagine typing a message and getting emoji suggestions that match the mood and meaning. This app does exactly that! It analyzes your text and suggests the best emojis.

## Project Parts

1. [Preparing Data and Training the Model](#part-1-preparing-data-and-training-the-model)
2. [Backend Server](#part-2-backend-server)
3. [Frontend App](#part-3-frontend-app)
4. [Deployment and Chrome Extension](#part-4-deployment-and-chrome-extension)

---

## Part 1: Preparing Data and Training the Model

This part cleans the data, sets up text processing, and trains a machine learning model to predict emojis.

### Files and Folders
```
data/
├── emojiclean.ipynb     # Notebook for cleaning and exploring emoji data
└── full_emoji.csv       # Raw emoji data with descriptions

trained_model/
├── emoji_lookup.json    # Maps emoji codes to names
├── training_metrics.json # How well the model performs
├── classifier.pkl       # The trained prediction model
├── tfidf_word.pkl       # Tool for turning words into numbers
├── tfidf_char.pkl       # Tool for turning characters into numbers
└── label_encoder.pkl    # Tool for handling emoji labels

train_model.py           # Script to train the model
evaluate_accuracy.py     # Tools to check model performance
test_predictions.py      # Script to test predictions
```

### Main Components

#### 1. Data Cleaning (`data/emojiclean.ipynb`)
- **Goal**: Get the emoji data ready for training
- **Steps**:
  - Remove duplicates and bad data
  - Make emoji names and descriptions consistent
  - Create better descriptions for training
  - Look at data patterns and distributions

#### 2. Text Processing (`nlp_pipeline.py` in backend/)
- **TextPreprocessor**: Cleans and breaks down text
  - Makes text lowercase, removes punctuation
  - Filters out common words
  - Simplifies words to their base form
- **FeatureExtractor**: Turns text into numbers for the computer
  - Uses TF-IDF to count important words and patterns
  - Finds common word combinations
- **SentimentAnalyzer**: Detects emotions in text
  - Analyzes positive/negative feelings
  - Classifies emotions like happy, sad, etc.

#### 3. Training the Model (`train_model.py`)
- **Combined Model**: Uses multiple machine learning methods together
  - Logistic Regression, SVM, Random Forest
  - Gives confidence scores for predictions
- **Backup System**: Uses advanced AI for predictions when needed
  - Uses a pre-trained language model
  - Pre-calculates emoji meanings for speed
- **Training Steps**:
  ```python
  # Load and clean data
  df = pd.read_csv('data/full_emoji.csv')
  preprocessor = TextPreprocessor()
  feature_extractor = FeatureExtractor()

  # Turn text into numbers
  X_word = tfidf_word.fit_transform(texts)
  X_char = tfidf_char.fit_transform(texts)
  X = hstack([X_word, X_char])

  # Train the combined model
  ensemble = ManualEnsemble([LogisticRegression(), SVC(probability=True), RandomForestClassifier()])
  ensemble.fit(X, y)
  ```

#### 4. Testing (`evaluate_accuracy.py`)
- **Performance Checks**:
  - Accuracy, precision, recall, F1-score
  - Top predictions accuracy
  - Error analysis
- **Validation**: Tests model on different data splits

---

## Part 2: Backend Server

A server built with FastAPI that provides the prediction service through web endpoints.

### Files and Folders
```
backend/
├── main.py              # Main server with API endpoints
├── nlp_pipeline.py      # Text processing tools
└── requirements.txt     # Python packages needed

api/
├── analyze.py           # Cloud function for analysis
├── index.py             # Cloud API entry point
├── predict.py           # Cloud prediction endpoint
├── requirements.txt     # Cloud packages
└── shared/
    ├── __init__.py
    ├── emoji_lookup.json
    └── nlp_engine.py     # Shared processing tools
```

### Main Components

#### 1. FastAPI Server (`backend/main.py`)
- **API Endpoints**:
  - `GET /api`: Info about the API and model
  - `POST /predict`: Main prediction service
  - `POST /analyze`: Detailed text analysis
  - `GET /metrics`: Model performance stats
- **Features**:
  - Allows cross-origin requests
  - Loads models automatically
  - Handles errors well
  - Processes requests quickly

#### 2. Prediction Process
```python
@app.post("/predict")
def predict_emoji(payload: dict):
    text = payload.get("text", "")
    top_k = payload.get("top_k", 2)

    # Clean the text
    processed = preprocessor.preprocess(text)

    # Use trained model first
    if USE_TRAINED_MODEL:
        results = _predict_with_trained_model(text, top_k)
    else:
        # Use AI model as backup
        results = _predict_with_transformer(text, top_k)

    return {
        "emojis": results,
        "model_used": "trained_classifier" if USE_TRAINED_MODEL else "transformer",
        "analysis": {
            "preprocessing": processed,
            "sentiment": sentiment_analyzer.full_analysis(text),
            "features": feature_extractor.get_top_ngrams(text)
        }
    }
```

#### 3. Cloud Deployment (`api/`)
- **Serverless Functions**: For cloud hosting
- **Shared Tools**: Reusable components
- **Same API**: Works like local server

#### 4. Model Loading
- **Trained Models**: Loads saved models
- **AI Setup**: Downloads and caches language models
- **Emoji Data**: Pre-calculated for speed
- **Fallback**: Tries different methods if one fails

---

## Part 3: Frontend App

A modern React app with real-time emoji prediction, built with Vite and styled with Tailwind CSS.

### Files and Folders
```
frontend/
├── public/
│   ├── manifest.json    # Chrome extension info
│   ├── _redirects       # Hosting redirects
│   └── vite.svg
├── src/
│   ├── App.jsx          # Main app component
│   ├── main.jsx         # App start point
│   ├── index.css        # Main styles
│   ├── App.css          # App styles
│   ├── components/
│   │   └── Navbar.jsx   # Navigation bar
│   └── pages/
│       ├── HomePage.jsx     # Welcome page
│       └── PredictorPage.jsx # Prediction page
├── package.json         # App dependencies
├── vite.config.js       # Build config
├── tailwind.config.js   # Style config
├── postcss.config.js    # CSS processing
└── eslint.config.js     # Code checking
```

### Main Components

#### 1. Real-Time Prediction (`PredictorPage.jsx`)
- **Smart Input**: Waits 500ms before predicting
- **Live Updates**: Shows suggestions as you type
- **Text Area**: Multi-line input
- **Loading**: Shows progress
- **Errors**: Friendly error messages

#### 2. User Interface
- **Modern Design**: Glass-like effects
- **Animations**: Smooth movements
- **Responsive**: Works on phone and desktop
- **Dark Theme**: Easy on eyes
- **Accessible**: Works with screen readers

#### 3. API Connection
```javascript
const debouncedPredict = useCallback(
    debounce(async (text) => {
        const response = await axios.post(`${API_BASE}/predict`, {
            text: text,
            top_k: 3
        });
        setPredictions(response.data.emojis);
        setAnalysis(response.data.analysis);
    }, 500),
    []
);
```

#### 4. Styling
- **Tailwind CSS**: Quick styling
- **Custom Parts**: Special buttons and panels
- **Colors**: Consistent theme
- **Motion**: Smooth animations

---

## Part 4: Deployment and Chrome Extension

Ways to deploy the app and set up a browser extension.

### Files and Folders
```
vercel.json              # Vercel hosting config
render.yaml              # Render hosting config
main.py                  # App launcher
requirements.txt         # Main packages

frontend/public/manifest.json  # Extension info
```

### Main Components

#### 1. App Launcher (`main.py`)
- **Full App**: Runs both server and frontend
- **Ports**: Manages connections
- **Building**: Compiles frontend
- **Development**: Auto-reloads on changes

#### 2. Chrome Extension (`manifest.json`)
```json
{
  "manifest_version": 3,
  "name": "Context-Aware Emoji Predictor",
  "action": {
    "default_popup": "index.html"
  },
  "host_permissions": ["http://localhost:8000/*"],
  "permissions": ["activeTab", "storage"]
}
```

- **Popup**: App as browser popup
- **Local Access**: Connects to local server
- **Ready for More**: Can add page features

#### 3. Deployment Options

##### Vercel (Cloud)
- **API Functions**: Serverless endpoints
- **Static Site**: Hosts the frontend
- **Settings**: API connection config

##### Render.com
- **Containers**: Docker deployment
- **Always On**: For backend
- **Scaling**: Grows with users

##### Local Development
- **Virtual Environment**: Isolated Python setup
- **Both Parts**: Server and frontend together
- **Auto-Reload**: Updates on code changes

#### 4. Build and Run
```bash
# Install packages
pip install -r requirements.txt
cd frontend && npm install

# Train model (optional)
python train_model.py

# Run everything
python main.py

# Or separate
python backend/main.py  # Server on :8000
cd frontend && npm run dev  # Frontend on :5173
```

#### 5. Extension Setup
- **Load Extension**: Use built frontend folder
- **Modern API**: Chrome extension standards
- **Permissions**: Only what's needed
- **Future**: Add more browser features

---

## How It Works

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   NLP Models    │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Saved/SC)    │
│                 │    │                 │    │                 │
│ • Live UI       │    │ • /predict       │    │ • Combined ML   │
│ • Smart input   │    │ • /analyze       │    │ • AI backup     │
│ • Animations    │    │ • CORS ok        │    │ • Embeddings    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Chrome         │
                    │  Extension      │
                    │  (Popup)        │
                    └─────────────────┘
```

## Technologies

- **Backend**: Python, FastAPI, scikit-learn, NLTK, AI models
- **Frontend**: React, Vite, Tailwind CSS, Framer Motion, Axios
- **ML/NLP**: TF-IDF, SVM, Logistic Regression, BERT
- **Deployment**: Vercel, Render, Docker
- **Extension**: Chrome Manifest V3

## Getting Started

1. **Setup**:
   ```bash
   git clone <repository>
   cd Context-Aware-Emoji-Prediction-from-Text-Using-NLP-Techniques
   pip install -r requirements.txt
   ```

2. **Train Model** (if needed):
   ```bash
   python train_model.py
   ```

3. **Run App**:
   ```bash
   python main.py
   ```

4. **Use It**:
   - Web App: http://localhost:8000
   - API Docs: http://localhost:8000/docs

5. **Chrome Extension**:
   - Build: `cd frontend && npm install && npm run build`
   - Load `frontend/dist/` as extension

## Contributing

We welcome help! Areas to improve:
- Better model accuracy
- More NLP features
- Better design
- Support more browsers
- Make it faster
