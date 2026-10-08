# Credit Risk Prediction — Render package

Repository structure:

    main.py
    credit_risk_model.pkl
    best_threshold.pkl
    requirements.txt
    render.yaml
    static/
        index.html
        style.css
        script.js

Your current FastAPI app mounts `static` at `/`:

    app.mount("/", StaticFiles(directory="static", html=True), name="static")

Therefore the HTML must load assets as `/style.css` and `/script.js`, not `/static/style.css` and `/static/script.js`.

The supplied frontend already uses the correct paths.

Render:
Build command:
    pip install -r requirements.txt

Start command:
    uvicorn main:app --host 0.0.0.0 --port $PORT

After deployment, test:
    https://YOUR-APP.onrender.com/
    https://YOUR-APP.onrender.com/style.css
    https://YOUR-APP.onrender.com/script.js

The two model files are not included in this package. Keep your existing
`credit_risk_model.pkl` and `best_threshold.pkl` in the repository root.
