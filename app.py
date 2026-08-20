import html
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rapidfuzz import fuzz, process


# Match threshold is intentionally near the top for easy tuning.
MATCH_THRESHOLD = 80
DEFAULT_MAX_GUESSES_PER_TEAM = 5
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
COMORBIDITIES_DIR = DATA_DIR / "comorbidities"
RARE_SYNDROMES_DIR = DATA_DIR / "rare_syndromes"
ASSETS_DIR = PROJECT_DIR / "assets"

DATASET_CONFIG = {
    "Obesity": {
        "file": "obesity_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "Genetic_Obesity_Comorbidity_Name",
        "score_col": "Number_of_Diseases",
    },
    "Cardiomyopathy": {
        "file": "cardiomyopathy_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "Genetic_Cardiomyopathy_Comorbidity_Name",
        "score_col": "Number_of_Diseases",
    },
    "Recurrent Respiratory Infections": {
        "file": "recurrent_respiratory_infection_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Visual Impairment": {
        "file": "visual_impairment_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Anemia": {
        "file": "anemia_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Scoliosis": {
        "file": "scoliosis_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Strabismus": {
        "file": "strabismus_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Arrhythmia": {
        "file": "arrhythmia_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Seizure": {
        "file": "seizure_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Ataxia": {
        "file": "ataxia_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Hypotonia": {
        "file": "hypotonia_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Polydactyly": {
        "file": "polydactyly_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Short Stature": {
        "file": "short_stature_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Diabetes": {
        "file": "diabetes_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Constipation": {
        "file": "constipation_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Myopia": {
        "file": "myopia_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
    "Intellectual Disability": {
        "file": "intellectual_disability_comorbidities_in_genetic_syndromes.csv",
        "finding_col": "co.name",
        "score_col": "count(co)",
    },
}
RARE_SYNDROMES_FILES = {
    "Obesity": "obesity_associated_rare_genetic_syndromes.csv",
    "Cardiomyopathy": "cardiomyopathy_associated_rare_genetic_syndromes.csv",
    "Recurrent Respiratory Infections": "recurrent_respiratory_infection_associated_rare_genetic_syndromes.csv",
    "Visual Impairment": "visual_impairment_associated_rare_genetic_syndromes.csv",
    "Anemia": "anemia_associated_rare_genetic_syndromes.csv",
    "Scoliosis": "scoliosis_associated_rare_genetic_syndromes.csv",
    "Strabismus": "strabismus_associated_rare_genetic_syndromes.csv",
    "Arrhythmia": "arrhythmia_associated_rare_genetic_syndromes.csv",
    "Seizure": "seizure_associated_rare_genetic_syndromes.csv",
    "Ataxia": "ataxia_associated_rare_genetic_syndromes.csv",
    "Hypotonia": "hypotonia_associated_rare_genetic_syndromes.csv",
    "Polydactyly": "polydactyly_associated_rare_genetic_syndromes.csv",
    "Short Stature": "short_stature_associated_rare_genetic_syndromes.csv",
    "Diabetes": "diabetes_associated_rare_genetic_syndromes.csv",
    "Constipation": "constipation_associated_rare_genetic_syndromes.csv",
    "Myopia": "myopia_associated_rare_genetic_syndromes.csv",
    "Intellectual Disability": "intellectual_disability_associated_rare_genetic_syndromes.csv",
}
FALLBACK_BRAND_IMAGE = ASSETS_DIR / "famhx_feud_logo.png"


def normalize_text(text: str) -> str:
    """Lowercase and trim whitespace for comparison."""
    return " ".join(text.lower().strip().split())


def clean_label(text: str) -> str:
    """Trim surrounding whitespace/quotes from CSV text values."""
    return str(text).strip().strip('"').strip("'").strip()


def detect_brand_image_path() -> Optional[Path]:
    """Find a project image to use for favicon/background."""
    top_level_images: List[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        top_level_images.extend(ASSETS_DIR.glob(pattern))
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        top_level_images.extend(PROJECT_DIR.glob(pattern))

    if top_level_images:
        keywords = ("famhx", "feud", "logo", "icon", "favicon", "steve", "harvey")

        def sort_key(image_path: Path) -> Tuple[int, float]:
            name = image_path.name.lower()
            has_keyword = any(keyword in name for keyword in keywords)
            return (0 if has_keyword else 1, -image_path.stat().st_mtime)

        return sorted(top_level_images, key=sort_key)[0]

    if FALLBACK_BRAND_IMAGE.exists():
        return FALLBACK_BRAND_IMAGE
    return None


def image_mime_type(image_path: Path) -> str:
    """Return MIME type for supported image formats."""
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def apply_landing_background(image_path: Path) -> None:
    """Apply a faded background image for the landing page."""
    try:
        image_bytes = image_path.read_bytes()
    except Exception:
        return

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    mime = image_mime_type(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: none;
    }}

    .stApp.famhx-landing-bg-enabled {{
        background-image: linear-gradient(rgba(255,255,255,0.82), rgba(255,255,255,0.82)),
                          url("data:{mime};base64,{encoded}");
        background-repeat: no-repeat;
        background-position: center 42%;
        background-size: min(70vw, 900px) auto;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    components.html(
        """
        <script>
        (function() {
          const parentDoc = window.parent.document;
          const app = parentDoc.querySelector('.stApp');
          if (!app) return;

          function parseRgb(colorValue) {
            const match = colorValue && colorValue.match(/\\d+/g);
            if (!match || match.length < 3) return [0, 0, 0];
            return [parseInt(match[0], 10), parseInt(match[1], 10), parseInt(match[2], 10)];
          }

          function isDarkModeByTextColor() {
            const [r, g, b] = parseRgb(window.parent.getComputedStyle(app).color);
            const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            return luminance > 180; // light text indicates dark theme
          }

          function syncLandingBackground() {
            const darkMode = isDarkModeByTextColor();
            app.classList.toggle('famhx-landing-bg-enabled', !darkMode);
          }

          syncLandingBackground();

          const observer = new MutationObserver(syncLandingBackground);
          observer.observe(parentDoc.documentElement, {
            attributes: true,
            subtree: true,
            attributeFilter: ['class', 'style', 'data-theme']
          });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def initialize_session_state() -> None:
    """Create all keys used by the game."""
    defaults = {
        "show_landing": True,
        "game_started": False,
        "presenting_finding": "Obesity",
        "team_names": ["Team 1", "Team 2"],
        "team_colors": ["#1f77b4", "#d62728"],
        "max_guesses_per_team": DEFAULT_MAX_GUESSES_PER_TEAM,
        "setup_max_guesses_per_team": DEFAULT_MAX_GUESSES_PER_TEAM,
        "scores": [0, 0],
        "guesses_used": [0, 0],
        "current_team_idx": 0,
        "matched_findings": set(),
        "team_correct": {0: [], 1: []},
        "team_guess_history": {0: [], 1: []},
        "df": None,
        "finding_col": None,
        "score_col": None,
        "official_to_score": {},
        "normalized_choices": [],
        "norm_to_official": {},
        "last_feedback": "",
        "last_feedback_type": "info",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_game_state() -> None:
    """Clear active game state while preserving loaded app."""
    st.session_state.show_landing = True
    st.session_state.game_started = False
    st.session_state.scores = [0, 0]
    st.session_state.guesses_used = [0, 0]
    st.session_state.current_team_idx = 0
    st.session_state.matched_findings = set()
    st.session_state.team_correct = {0: [], 1: []}
    st.session_state.team_guess_history = {0: [], 1: []}
    st.session_state.df = None
    st.session_state.finding_col = None
    st.session_state.score_col = None
    st.session_state.official_to_score = {}
    st.session_state.normalized_choices = []
    st.session_state.norm_to_official = {}
    st.session_state.last_feedback = ""
    st.session_state.last_feedback_type = "info"
    st.session_state.setup_max_guesses_per_team = st.session_state.max_guesses_per_team


def reset_round_for_setup() -> None:
    """Reset game round and return to setup while keeping chosen settings."""
    st.session_state.show_landing = False
    st.session_state.game_started = False
    st.session_state.scores = [0, 0]
    st.session_state.guesses_used = [0, 0]
    st.session_state.current_team_idx = 0
    st.session_state.matched_findings = set()
    st.session_state.team_correct = {0: [], 1: []}
    st.session_state.team_guess_history = {0: [], 1: []}
    st.session_state.df = None
    st.session_state.finding_col = None
    st.session_state.score_col = None
    st.session_state.official_to_score = {}
    st.session_state.normalized_choices = []
    st.session_state.norm_to_official = {}
    st.session_state.last_feedback = ""
    st.session_state.last_feedback_type = "info"
    st.session_state.setup_max_guesses_per_team = st.session_state.max_guesses_per_team


def load_dataset(presenting_finding: str) -> bool:
    """Load and validate the selected CSV. Returns True on success."""
    config = DATASET_CONFIG[presenting_finding]
    file_name = config["file"]
    file_path = COMORBIDITIES_DIR / file_name
    finding_col = config["finding_col"]
    score_col = config["score_col"]

    if not file_path.exists():
        st.error(f"Required file not found: `{file_path}`")
        return False

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        st.error(f"Could not read `{file_path}`: {exc}")
        return False

    missing_cols = [col for col in [finding_col, score_col] if col not in df.columns]
    if missing_cols:
        st.error(
            "CSV columns are not as expected. Missing: "
            + ", ".join(f"`{col}`" for col in missing_cols)
        )
        return False

    clean_df = df[[finding_col, score_col]].copy()
    clean_df[finding_col] = clean_df[finding_col].apply(clean_label)
    clean_df = clean_df[clean_df[finding_col] != ""]
    clean_df[score_col] = pd.to_numeric(clean_df[score_col], errors="coerce").fillna(0).astype(int)

    official_to_score = dict(zip(clean_df[finding_col], clean_df[score_col]))
    norm_to_official: Dict[str, str] = {}
    for official_name in official_to_score.keys():
        norm_to_official[normalize_text(official_name)] = official_name

    st.session_state.df = clean_df
    st.session_state.finding_col = finding_col
    st.session_state.score_col = score_col
    st.session_state.official_to_score = official_to_score
    st.session_state.norm_to_official = norm_to_official
    st.session_state.normalized_choices = list(norm_to_official.keys())
    return True


def max_guesses_per_team() -> int:
    """Return configured guesses per team for current game."""
    return int(st.session_state.get("max_guesses_per_team", DEFAULT_MAX_GUESSES_PER_TEAM))


def winner_team_idx() -> Optional[int]:
    """Return winner team index when game is complete; None for tie/incomplete."""
    if not all_guesses_completed():
        return None
    score_0, score_1 = st.session_state.scores
    if score_0 == score_1:
        return None
    return 0 if score_0 > score_1 else 1


def game_over_feedback() -> Tuple[str, str]:
    """Build final game-over message for feedback banner."""
    winner_idx = winner_team_idx()
    if winner_idx is None:
        team_1, team_2 = st.session_state.team_names
        score_1, score_2 = st.session_state.scores
        return (f"🏁 Game over. It's a tie! {team_1} {score_1} - {score_2} {team_2}.", "info")

    winner_name = st.session_state.team_names[winner_idx]
    return (f"🏁 Game over. {winner_name} wins! 🎉", "success")


def all_guesses_completed() -> bool:
    return all(used >= max_guesses_per_team() for used in st.session_state.guesses_used)


def next_available_team(start_idx: int) -> int:
    """Find next team that still has guesses left."""
    max_guesses = max_guesses_per_team()
    for offset in range(1, 3):
        idx = (start_idx + offset) % 2
        if st.session_state.guesses_used[idx] < max_guesses:
            return idx
    return start_idx


def log_guess(
    team_idx: int, guess: str, matched_finding: str, points: int, result: str
) -> None:
    """Store one guess attempt for history display."""
    display_guess = guess.strip() or "(empty guess)"
    st.session_state.team_guess_history[team_idx].append(
        {
            "Team": st.session_state.team_names[team_idx],
            "Team Color": st.session_state.team_colors[team_idx],
            "Guess": display_guess,
            "Matched Finding": matched_finding,
            "Points": points,
            "Result": result,
        }
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.22) -> str:
    """Convert #RRGGBB to rgba(...) for soft row highlighting."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return "rgba(128, 128, 128, 0.15)"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_combined_history_df() -> pd.DataFrame:
    """Combine both team histories and sort by points descending."""
    all_rows: List[Dict[str, object]] = (
        st.session_state.team_guess_history[0] + st.session_state.team_guess_history[1]
    )
    if not all_rows:
        return pd.DataFrame(columns=["Team", "Guess", "Matched Finding", "Points", "Result"])

    history_df = pd.DataFrame(all_rows)
    history_df["Points"] = pd.to_numeric(history_df["Points"], errors="coerce").fillna(0).astype(int)
    history_df = history_df.sort_values(
        by=["Points", "Team", "Guess"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return history_df


def render_combined_history_table() -> None:
    """Render all team guesses in one color-coded table."""
    history_df = build_combined_history_df()
    if history_df.empty:
        st.write("No guesses yet.")
        return

    table_df = history_df.rename(
        columns={
            "Guess": "Syndromic Finding",
            "Points": "Number of Genetic Diseases",
        }
    )[["Syndromic Finding", "Number of Genetic Diseases", "Result"]]
    table_df["Number of Genetic Diseases"] = table_df.apply(
        lambda row: row["Number of Genetic Diseases"] if row["Result"] == "Scored" else "-",
        axis=1,
    )
    table_df = table_df[["Syndromic Finding", "Number of Genetic Diseases"]]

    def row_style(row: pd.Series) -> List[str]:
        background = hex_to_rgba(str(history_df.loc[row.name, "Team Color"]))
        return [f"background-color: {background}"] * len(row)

    styled = table_df.style.apply(row_style, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_team_score_cards() -> None:
    """Render team score cards tinted with each team's selected color."""
    max_guesses = max_guesses_per_team()
    game_in_progress = not all_guesses_completed()
    active_team_idx = st.session_state.current_team_idx
    winner_idx = winner_team_idx()
    card_cols = st.columns(2)
    for idx, card_col in enumerate(card_cols):
        with card_col:
            team_name = st.session_state.team_names[idx]
            safe_team_name = html.escape(team_name)
            team_color = st.session_state.team_colors[idx]
            is_active = game_in_progress and idx == active_team_idx
            is_winner = (not game_in_progress) and winner_idx is not None and idx == winner_idx
            is_tie = (not game_in_progress) and winner_idx is None
            bg = hex_to_rgba(team_color, alpha=0.26 if is_active else 0.14)
            score = st.session_state.scores[idx]
            guesses = st.session_state.guesses_used[idx]
            border_width = "2px" if (is_active or is_winner) else "1px"
            shadow = f"0 0 0 1px {team_color}55" if (is_active or is_winner) else "none"
            opacity = "1.0" if (is_active or is_winner or is_tie) else "0.82"
            status_badge = ""
            if is_active:
                status_badge = (
                    "<div style=\"font-size: 0.85rem; margin-top: 4px; font-weight: 600;\">"
                    f"🎲 {safe_team_name}'s turn</div>"
                )
            elif is_winner:
                status_badge = (
                    "<div style=\"font-size: 0.85rem; margin-top: 4px; font-weight: 600;\">"
                    f"🎉 {safe_team_name} wins</div>"
                )
            elif is_tie:
                status_badge = (
                    "<div style=\"font-size: 0.85rem; margin-top: 4px; font-weight: 600;\">"
                    "🤝 Tie game</div>"
                )
            guesses_line = ""
            if game_in_progress:
                guesses_line = (
                    "<div style=\"font-size: 0.9rem; margin-top: 6px;\">"
                    f"Guesses used: {guesses}/{max_guesses}</div>"
                )
            card_html = (
                "<div style=\""
                f"border: {border_width} solid {team_color};"
                "border-radius: 10px;"
                "padding: 14px 16px;"
                f"background: {bg};"
                "margin-bottom: 8px;"
                f"box-shadow: {shadow};"
                f"opacity: {opacity};"
                "\">"
                f"<div style=\"font-size: 0.95rem; font-weight: 700; color: {team_color};\">"
                f"{safe_team_name}</div>"
                f"{status_badge}"
                "<div style=\"font-size: 2rem; font-weight: 700; line-height: 1.1; margin-top: 4px;\">"
                f"{score}</div>"
                f"{guesses_line}"
                "</div>"
            )
            st.markdown(
                card_html,
                unsafe_allow_html=True,
            )


def score_guess(guess: str) -> Tuple[str, str]:
    """Process one guess for current team and return feedback and message type."""
    team_idx = st.session_state.current_team_idx
    team_name = st.session_state.team_names[team_idx]

    if st.session_state.guesses_used[team_idx] >= max_guesses_per_team():
        st.session_state.current_team_idx = next_available_team(team_idx)
        return (
            f"{team_name} has already used all guesses. Turn moved to the other team.",
            "info",
        )

    normalized_guess = normalize_text(guess)
    st.session_state.guesses_used[team_idx] += 1

    if not normalized_guess:
        log_guess(
            team_idx=team_idx,
            guess=guess,
            matched_finding="No match found",
            points=0,
            result="No points",
        )
        st.session_state.current_team_idx = next_available_team(team_idx)
        return ("😢 No match found in Top 100. 0 points.", "error")

    match: Optional[Tuple[str, float, int]] = process.extractOne(
        normalized_guess,
        st.session_state.normalized_choices,
        scorer=fuzz.WRatio,
    )

    if not match:
        log_guess(
            team_idx=team_idx,
            guess=guess,
            matched_finding="No match found",
            points=0,
            result="No points",
        )
        st.session_state.current_team_idx = next_available_team(team_idx)
        return ("😢 No match found in Top 100. 0 points.", "error")

    matched_normalized, similarity, _ = match
    if similarity < MATCH_THRESHOLD:
        log_guess(
            team_idx=team_idx,
            guess=guess,
            matched_finding="No match found",
            points=0,
            result="No points",
        )
        st.session_state.current_team_idx = next_available_team(team_idx)
        return ("😢 No match found in Top 100. 0 points.", "error")

    official_name = st.session_state.norm_to_official[matched_normalized]
    if official_name in st.session_state.matched_findings:
        log_guess(
            team_idx=team_idx,
            guess=guess,
            matched_finding=official_name,
            points=0,
            result="Already guessed",
        )
        st.session_state.current_team_idx = next_available_team(team_idx)
        return (
            f"Already guessed: {official_name}. No additional points awarded.",
            "warning",
        )

    points = int(st.session_state.official_to_score.get(official_name, 0))
    st.session_state.matched_findings.add(official_name)
    st.session_state.scores[team_idx] += points
    st.session_state.team_correct[team_idx].append((official_name, points))
    log_guess(
        team_idx=team_idx,
        guess=guess,
        matched_finding=official_name,
        points=points,
        result="Scored",
    )
    st.session_state.current_team_idx = next_available_team(team_idx)
    return (f"Correct! {team_name} earns {points} points.", "success")


def render_game_board() -> None:
    """Render active game view."""
    presenting = st.session_state.presenting_finding
    st.markdown(
        "### Syndromic Findings Associated with Genetic "
        f"<span style='font-weight:800; text-decoration: underline;'>{presenting}</span>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        render_team_score_cards()

        if not all_guesses_completed():
            with st.form("guess_form", clear_on_submit=True):
                guess_text = st.text_input("Enter guess", placeholder="Type a syndromic finding...")
                submitted = st.form_submit_button("Submit Guess")
                if submitted:
                    feedback, feedback_type = score_guess(guess_text)
                    st.session_state.last_feedback = feedback
                    st.session_state.last_feedback_type = feedback_type
                    # Force a fresh render so score/turn widgets reflect updated state immediately.
                    st.rerun()
        else:
            final_feedback, final_feedback_type = game_over_feedback()
            st.session_state.last_feedback = final_feedback
            st.session_state.last_feedback_type = final_feedback_type

        if st.session_state.last_feedback:
            if st.session_state.last_feedback_type == "error":
                st.error(st.session_state.last_feedback)
            elif st.session_state.last_feedback_type == "warning":
                st.warning(st.session_state.last_feedback)
            elif st.session_state.last_feedback_type == "success":
                st.success(st.session_state.last_feedback)
            else:
                st.info(st.session_state.last_feedback)

    with right_col:
        render_combined_history_table()

    if not all_guesses_completed():
        return


def render_endgame() -> None:
    """Render final results and reference table."""
    if st.button("Play Again", type="primary", key="play_again_button"):
        reset_round_for_setup()
        st.rerun()

    st.write(
        "### Top 100 Syndromic Findings Associated with Genetic "
        f"{st.session_state.presenting_finding}"
    )
    guessed_finding_to_color: Dict[str, str] = {}
    for team_idx in [0, 1]:
        team_color = st.session_state.team_colors[team_idx]
        for finding, _points in st.session_state.team_correct[team_idx]:
            guessed_finding_to_color[finding] = team_color

    top100 = (
        st.session_state.df.sort_values(by=st.session_state.score_col, ascending=False)
        .head(100)
        .reset_index(drop=True)
    )
    top100_display = top100.rename(
        columns={
            st.session_state.finding_col: "Syndromic Finding",
            st.session_state.score_col: "Number of Genetic Diseases",
        }
    )

    def highlight_guessed_rows(row: pd.Series) -> List[str]:
        finding = str(row["Syndromic Finding"])
        team_color = guessed_finding_to_color.get(finding)
        if not team_color:
            return [""] * len(row)
        background = hex_to_rgba(team_color, alpha=0.24)
        return [f"background-color: {background}"] * len(row)

    styled_top100 = top100_display.style.apply(highlight_guessed_rows, axis=1)
    st.dataframe(styled_top100, use_container_width=True, hide_index=True, height=420)

    presenting = st.session_state.presenting_finding
    rare_file = RARE_SYNDROMES_FILES.get(presenting)
    if not rare_file:
        st.warning(f"No rare-syndrome dataset is configured for `{presenting}`.")
        return

    rare_path = RARE_SYNDROMES_DIR / rare_file
    st.write(f"### {presenting}-Associated Rare Genetic Syndromes")
    if not rare_path.exists():
        st.warning(f"Optional file not found: `{rare_path}`")
        return

    try:
        rare_df = pd.read_csv(rare_path)
    except Exception as exc:
        st.error(f"Could not read `{rare_path}`: {exc}")
        return

    first_col = rare_df.columns[0]
    rare_df[first_col] = rare_df[first_col].apply(clean_label)
    rare_df = rare_df[rare_df[first_col] != ""].sort_values(
        by=first_col,
        key=lambda s: s.str.lower(),
    ).reset_index(drop=True)

    st.caption(f"Total diseases listed: **{len(rare_df)}**")
    styled_rare = rare_df.style.hide(axis="columns")
    st.dataframe(styled_rare, use_container_width=True, hide_index=True, height=420)


def render_landing_page(brand_image_path: Optional[Path]) -> None:
    """Render welcome/intro screen before setup."""
    if brand_image_path:
        apply_landing_background(brand_image_path)

    st.title("FamHx Feud")
    st.write("### Welcome to FamHx Feud!")
    st.write(
        "Many genetic syndromes present first as familiar clinical problems. "
        "In this game, you will choose a presenting finding, such as obesity, "
        "cardiomyopathy, or seizures, and your team will guess other findings "
        "which would make this start to look like a genetic syndrome."
    )
    st.write("### How it works")
    st.write("Teams take turns guessing associated findings.")
    st.write(
        "Correct guesses earn points based on how many genetic diseases "
        "share that finding."
    )
    st.write(
        "The goal is not to memorize rare diseases. It is to build intuition for when "
        "a common presentation may be part of a broader syndrome."
    )

    if st.button("Start", type="primary"):
        st.session_state.show_landing = False
        st.rerun()

    st.caption(
        "Data were obtained from the Human Phenotype Ontology, Orphanet and the "
        "HPO-ORDO Ontology Mapping (HOOM) databases in 2022. Game designed and built "
        "by Tim McLerran, DO."
    )


def main() -> None:
    brand_image_path = detect_brand_image_path()
    page_icon: object = ":stethoscope:"
    if brand_image_path:
        page_icon = str(brand_image_path)

    st.set_page_config(page_title="FamHx Feud", page_icon=page_icon)
    initialize_session_state()

    if st.session_state.show_landing:
        render_landing_page(brand_image_path)
        return

    if not st.session_state.game_started:
        st.title("FamHx Feud")
        st.write("### Setup")
        setup_cols = st.columns(2)
        with setup_cols[0]:
            team1 = st.text_input("Team 1 Name", value=st.session_state.team_names[0]).strip()
            team1_color = st.color_picker(
                "Team 1 Color",
                value=st.session_state.team_colors[0],
            )
        with setup_cols[1]:
            team2 = st.text_input("Team 2 Name", value=st.session_state.team_names[1]).strip()
            team2_color = st.color_picker(
                "Team 2 Color",
                value=st.session_state.team_colors[1],
            )
        presenting_options = sorted(DATASET_CONFIG.keys(), key=str.lower)
        presenting = st.selectbox(
            "Choose presenting finding",
            options=presenting_options,
            index=presenting_options.index(st.session_state.presenting_finding),
        )
        setup_guess_count = st.number_input(
            "Guesses per team",
            min_value=1,
            max_value=25,
            step=1,
            value=int(
                st.session_state.get(
                    "setup_max_guesses_per_team",
                    DEFAULT_MAX_GUESSES_PER_TEAM,
                )
            ),
            key="setup_max_guesses_per_team",
            help="Choose how many guesses each team gets in this game.",
        )

        if st.button("Start Game", type="primary"):
            st.session_state.team_names = [team1 or "Team 1", team2 or "Team 2"]
            st.session_state.team_colors = [team1_color, team2_color]
            st.session_state.max_guesses_per_team = int(setup_guess_count)
            st.session_state.presenting_finding = presenting
            st.session_state.scores = [0, 0]
            st.session_state.guesses_used = [0, 0]
            st.session_state.current_team_idx = 0
            st.session_state.matched_findings = set()
            st.session_state.team_correct = {0: [], 1: []}
            st.session_state.team_guess_history = {0: [], 1: []}
            st.session_state.last_feedback = ""
            st.session_state.last_feedback_type = "info"

            if load_dataset(presenting):
                st.session_state.game_started = True
                st.rerun()

    else:
        render_game_board()
        if all_guesses_completed():
            render_endgame()


if __name__ == "__main__":
    main()
