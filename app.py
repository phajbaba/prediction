import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. DATA INGESTION (LIVE & FREE)
# ==========================================
LEAGUES = {
    "English Premier League": "E0",
    "Spanish La Liga": "SP1",
    "German Bundesliga": "D1",
    "Italian Serie A": "I1",
    "French Ligue 1": "F1"
}

SEASONS = {
    "2026/2027": "2627",
    "2025/2026": "2526",
    "2024/2025": "2425",
    "2023/2024": "2324",
    "2022/2023": "2223"
}

@st.cache_data(ttl=3600)
def load_data(league_code, season_code):
    """Fetches real match data directly from football-data.co.uk."""
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    try:
        # Some older files have encoding differences
        df = pd.read_csv(url, encoding='unicode_escape')
        # Keep only essential columns to avoid clutter
        df = df[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
        df = df.dropna()
        return df
    except Exception as e:
        return pd.DataFrame()

# ==========================================
# 2. STATISTICAL ENGINE (POISSON MODEL)
# ==========================================
def calculate_team_ratings(df):
    """Calculates Attack and Defense strength for all teams based on goals scored/conceded."""
    if df.empty:
        return None, None
    
    # League Averages
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
    
    # Calculate Team Stats
    home_stats = df.groupby('HomeTeam').agg({'FTHG': 'mean', 'FTAG': 'mean'}).rename(
        columns={'FTHG': 'HomeScored', 'FTAG': 'HomeConceded'})
    away_stats = df.groupby('AwayTeam').agg({'FTAG': 'mean', 'FTHG': 'mean'}).rename(
        columns={'FTAG': 'AwayScored', 'FTHG': 'AwayConceded'})
    
    teams = pd.concat([home_stats, away_stats], axis=1).fillna(0)
    
    # Strength Ratings (Team Avg / League Avg) - added 0.01 to prevent division by zero
    teams['HomeAttack'] = teams['HomeScored'] / max(avg_home_goals, 0.01)
    teams['HomeDefense'] = teams['HomeConceded'] / max(avg_away_goals, 0.01)
    teams['AwayAttack'] = teams['AwayScored'] / max(avg_away_goals, 0.01)
    teams['AwayDefense'] = teams['AwayConceded'] / max(avg_home_goals, 0.01)
    
    return teams, (avg_home_goals, avg_away_goals)

def simulate_match(home_xg, away_xg, max_goals=8):
    """Generates a matrix of exact score probabilities."""
    # Create Poisson distribution arrays for home and away
    home_probs = poisson.pmf(np.arange(max_goals), home_xg)
    away_probs = poisson.pmf(np.arange(max_goals), away_xg)
    
    # Outer product gives the exact score matrix (row = home goals, col = away goals)
    matrix = np.outer(home_probs, away_probs)
    
    # Calculate Markets
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    
    u15 = matrix[0,0] + matrix[1,0] + matrix[0,1]
    o15 = 1 - u15
    u25 = u15 + matrix[1,1] + matrix[2,0] + matrix[0,2]
    o25 = 1 - u25
    u35 = u25 + matrix[3,0] + matrix[2,1] + matrix[1,2] + matrix[0,3]
    o35 = 1 - u35
    
    gg = np.sum(matrix[1:, 1:])
    ng = 1 - gg
    
    # Top 3 Exact Scores
    flat = matrix.flatten()
    top_3_idx = flat.argsort()[-3:][::-1]
    top_scores = [(idx // max_goals, idx % max_goals, flat[idx]) for idx in top_3_idx]
    
    return {
        "1X2": {"Home": home_win, "Draw": draw, "Away": away_win},
        "DoubleChance": {"1X": home_win + draw, "12": home_win + away_win, "X2": draw + away_win},
        "Goals": {"Over 1.5": o15, "Under 1.5": u15, "Over 2.5": o25, "Under 2.5": u25, "Over 3.5": o35},
        "BTTS": {"GG": gg, "NG": ng},
        "Exact": top_scores
    }

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Pro Football Predictor", layout="wide")
st.title("📊 Statistical Match Predictor")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("Match Setup")
selected_league = st.sidebar.selectbox("Select League", list(LEAGUES.keys()))
selected_season = st.sidebar.selectbox("Select Season", list(SEASONS.keys()))

league_code = LEAGUES[selected_league]
season_code = SEASONS[selected_season]

# Load Dataset
df = load_data(league_code, season_code)

if df.empty:
    st.error(f"Data not available yet for {selected_league} ({selected_season}). Choose an older season.")
else:
    teams_data, league_avgs = calculate_team_ratings(df)
    team_list = sorted(df['HomeTeam'].unique())
    
    home_team = st.sidebar.selectbox("Home Team", team_list, index=0)
    away_team = st.sidebar.selectbox("Away Team", team_list, index=1 if len(team_list) > 1 else 0)
    
    if st.sidebar.button("Predict Match"):
        if home_team == away_team:
            st.warning("Please select different teams.")
        else:
            # Calculate Expected Goals (xG) based on Poisson metrics
            home_xg = teams_data.loc[home_team, 'HomeAttack'] * teams_data.loc[away_team, 'AwayDefense'] * league_avgs[0]
            away_xg = teams_data.loc[away_team, 'AwayAttack'] * teams_data.loc[home_team, 'HomeDefense'] * league_avgs[1]
            
            # Run Simulation
            results = simulate_match(home_xg, away_xg)
            
            # --- DISPLAY RESULTS ---
            st.markdown(f"### 🏆 {home_team} vs {away_team}")
            st.caption(f"**Expected Goals (xG):** {home_team} **{home_xg:.2f}** | {away_team} **{away_xg:.2f}**")
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            # 1X2 Market
            with col1:
                st.subheader("Match Winner (1X2)")
                st.write(f"**Home ({home_team}):** {results['1X2']['Home'] * 100:.1f}%")
                st.write(f"**Draw (X):** {results['1X2']['Draw'] * 100:.1f}%")
                st.write(f"**Away ({away_team}):** {results['1X2']['Away'] * 100:.1f}%")
                
                st.markdown("---")
                st.subheader("Double Chance")
                st.write(f"**1X (Home/Draw):** {results['DoubleChance']['1X'] * 100:.1f}%")
                st.write(f"**X2 (Draw/Away):** {results['DoubleChance']['X2'] * 100:.1f}%")
            
            # Goals Market
            with col2:
                st.subheader("Goals Markets")
                st.write(f"**Over 1.5:** {results['Goals']['Over 1.5'] * 100:.1f}%")
                st.write(f"**Over 2.5:** {results['Goals']['Over 2.5'] * 100:.1f}%")
                st.write(f"**Under 2.5:** {results['Goals']['Under 2.5'] * 100:.1f}%")
                st.write(f"**Over 3.5:** {results['Goals']['Over 3.5'] * 100:.1f}%")
                
                st.markdown("---")
                st.subheader("Both Teams to Score")
                st.write(f"**Yes (GG):** {results['BTTS']['GG'] * 100:.1f}%")
                st.write(f"**No (NG):** {results['BTTS']['NG'] * 100:.1f}%")
            
            # Top Predictions & Value
            with col3:
                st.subheader("Most Likely Exact Scores")
                for h_goals, a_goals, prob in results['Exact']:
                    st.write(f"**{h_goals} - {a_goals}** ➔ {prob * 100:.1f}%")
                
                st.markdown("---")
                st.subheader("🤖 AI Best Picks")
                # Logic to determine the safest/highest probability bets
                best_picks = []
                if results['1X2']['Home'] > 0.60: best_picks.append("Home Win (1)")
                elif results['1X2']['Away'] > 0.60: best_picks.append("Away Win (2)")
                elif results['DoubleChance']['1X'] > 0.75: best_picks.append("Double Chance 1X")
                elif results['DoubleChance']['X2'] > 0.75: best_picks.append("Double Chance X2")
                
                if results['Goals']['Over 1.5'] > 0.75: best_picks.append("Over 1.5 Goals")
                if results['Goals']['Over 2.5'] > 0.55: best_picks.append("Over 2.5 Goals")
                if results['BTTS']['GG'] > 0.55: best_picks.append("BTTS: Yes (GG)")
                
                if best_picks:
                    for pick in best_picks[:3]: # Show top 3 safest picks
                        st.success(f"🔥 {pick}")
                else:
                    st.warning("No highly confident markets found. Skip this match.")