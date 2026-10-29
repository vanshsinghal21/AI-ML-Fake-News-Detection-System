# Truthlens — Fake News Detection
Truthlens is a web application for checking news text with a machine-learning model. It uses a Flask backend and a web frontend to display a prediction. LIME and SHAP can be used to help explain model predictions when they are configured in the application.

> **Note:** A model prediction is not definitive proof that a news story is true or false. Verify important claims using reliable sources.

## Features

- Submit news text through a web interface.
- Display the model's prediction.
- Explain predictions with LIME and/or SHAP, if enabled in the current implementation.
- Flask backend with HTML/CSS/JavaScript frontend.
- Can be deployed as a web service using Render.

## Project Structure

Your exact filenames may differ. Keep the model files and paths consistent with those used in `app.py`.

```text
fake-news-detection/
├── fake news/
│   ├── app.py
│   ├── requirements.txt
│   ├── templates/
│   │   └── index.html
│   ├── static/
│   │   └── style.css
│   └── <saved model / vectorizer files>
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10 or a version supported by your installed dependencies
- pip
- Git (for GitHub deployment)

Python packages should be listed in `requirements.txt`. Depending on the code, these may include Flask, Gunicorn, joblib, scikit-learn, LIME, and SHAP.

## Run Locally on Windows

Open PowerShell in the folder containing `app.py`.

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this for the current terminal session, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` does not list all imported packages, add the required packages to it and install them. For example:

```powershell
python -m pip install flask joblib scikit-learn lime shap gunicorn
```

### 3. Start the Flask application

```powershell
python app.py
```

Open the local URL printed in the terminal, commonly `http://127.0.0.1:5000`.

## Deploy to GitHub and Render

### 1. Push the project to GitHub

Create a GitHub repository, then run these commands from the repository's root folder. If Git is already initialized or a remote already exists, do not repeat those setup commands.

```powershell
git status
git add .
git commit -m "Prepare SatyaCheck for deployment"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fake-news-detection.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. If `origin` already exists, check it with `git remote -v` and update it only if necessary.

### 2. Configure Render

Create a **Web Service** in Render and connect the GitHub repository.

Use these settings only if `app.py` and `requirements.txt` are both inside the `fake news` folder:

- **Root Directory:** `fake news`
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`

The start command assumes the Flask application object inside `app.py` is named `app`. If your file layout or Flask object name differs, update the settings accordingly.

Ensure that every model/vectorizer file loaded by `app.py` is included in the repository and is accessible from the service's configured root directory. Do not commit `.venv`, credentials, `.env` files, or private data.

After a successful deployment, Render provides a public URL for the website.

## Update the Frontend

When you edit HTML, CSS, or JavaScript, save the changes and push them to GitHub:

```powershell
git add .
git commit -m "Update frontend"
git push origin main
```

If automatic deployment is enabled in Render, it will deploy the new commit. A frontend-only change normally does not require retraining the model. Test the deployed website after each update.

## Troubleshooting

### `ModuleNotFoundError: No module named 'lime'` or `'shap'`

Activate the project's virtual environment and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Make sure the missing package is listed in `requirements.txt` so that Render installs it too.

### `can't open file 'app.py'`

Change directory to the folder that actually contains `app.py`, then run `python app.py`.

### Model file not found

Check the filename and path used by `joblib.load()` or other model-loading code. Ensure the required file is committed to GitHub and available at the expected path in Render.

### Deployment fails or the site does not start

Review the Render build and runtime logs. Confirm that the root directory, dependencies, start command, model paths, and Python compatibility are correct.

## Limitations and Responsible Use

- Predictions depend on the training data, model quality, and input text.
- A prediction should be treated as an indicator, not a verified fact.
- Avoid submitting private or sensitive information.
- Do not claim a story is definitively fake or genuine based only on a model score.


## License

Add a license here if you intend to distribute the project publicly. Do not claim a license unless you have selected one.
