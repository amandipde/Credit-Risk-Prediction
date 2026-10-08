# Credit Risk Prediction UI

`main.py` is NOT included or modified in this package.

Put these files alongside your existing `main.py`:

```text
your-project/
├── main.py
├── credit_risk_model.pkl
├── best_threshold.pkl
├── requirements.txt
├── render.yaml
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

## Local

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Because the existing `main.py` serves `/` as `{"message": "Hello, World!"}`, open the UI at:

http://127.0.0.1:8000/static/index.html

The API docs remain at:

http://127.0.0.1:8000/docs

The frontend calls `/predict/`, so it uses the same API origin.

## Render

The included `render.yaml` uses:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Keep your existing `main.py`, `credit_risk_model.pkl`, and `best_threshold.pkl` in the repository root.

For model pickle compatibility, pin the exact scikit-learn and xgboost versions used to train the model.
