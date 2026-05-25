# FamHx Feud

Local Streamlit teaching-session prototype for a two-team clinical genetics game.

## Run locally (recommended with virtualenv)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## How it works

- Choose a presenting finding.
- Enter two team names and start the game.
- Teams alternate turns with a configurable number of guesses per team.
- Correct guesses are fuzzy-matched and scored using the dataset's disease-count field.
- Duplicate findings cannot score twice.
- Endgame shows winner state, top 100 findings from the selected dataset, and rare syndromes.

## Project layout

- `app.py` - Streamlit app
- `data/comorbidities/` - presenting-finding comorbidity CSVs
- `data/rare_syndromes/` - rare-syndrome CSVs used in endgame
- `assets/` - branding image(s) used for favicon/landing page
