from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
import statsmodels.api as sm


import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')


def calculate_sets(x, i):
    sets = 0
    for e in x.split(' '):
        split = e.split('-')
        sets += int(split[i])
    return sets


def csv_to_sql():
    data = pd.read_csv("atp_tennis.csv")
    data.dropna(inplace=True)
    for x in data.index:
        if (data.loc[x, 'Rank_1'] <= 0 or
                data.loc[x, 'Rank_2'] <= 0 or
                data.loc[x, 'Pts_1'] <= 0 or
                data.loc[x, 'Pts_2'] <= 0 or
                data.loc[x, 'Odd_1'] <= 0 or
                data.loc[x, 'Odd_2'] <= 0):
            data.drop(x, inplace=True)
    data['Player_1_games'] = data['Score'].apply(lambda x: calculate_sets(x, 0))
    data['Player_2_games'] = data['Score'].apply(lambda x: calculate_sets(x, 1))
    conn = sqlite3.connect('tennis_data.db')
    data.to_sql('Matches', conn, if_exists='replace', index=False)
    print("Dane zapisane do sql")
    conn.close()


def rank_wins_plot(data):
    ranking_wins_1 = data[data['Winner'] == data['Player_1']].groupby("Rank_1").size().reset_index(
        name="Wins").sort_values(
        by='Rank_1').rename(columns={"Rank_1": "Rank"})
    ranking_wins_2 = data[data['Winner'] == data['Player_2']].groupby("Rank_2").size().reset_index(
        name="Wins").sort_values(
        by='Rank_2').rename(columns={"Rank_2": "Rank"})
    ranking_wins = pd.concat([ranking_wins_1, ranking_wins_2])
    ranking_wins = ranking_wins.groupby("Rank")['Wins'].sum().reset_index().sort_values(by='Rank')
    print(ranking_wins)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Rank', y='Wins', data=ranking_wins, color='blue')
    # Dodanie tytułu i etykiet osi
    plt.title('Liczba wygranych meczów w zależności od pozycji w rankingu', fontsize=16)
    plt.xlabel('Pozycja w rankingu', fontsize=12)
    plt.ylabel('Liczba wygranych meczów', fontsize=12)
    # Ograniczenie widocznych wartości osi X (np. do top 50 pozycji)
    plt.xticks(rotation=90)
    plt.xlim(0, 50)  # Wyświetlanie tylko pierwszych 50 pozycji w rankingu
    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()


def rankPoints_wins_plot(data):
    points_wins_1 = data[data['Winner'] == data['Player_1']].groupby("Pts_1").size().reset_index(
        name="Wins").sort_values(
        by='Pts_1').rename(columns={"Pts_1": "Pts"})
    points_wins_2 = data[data['Winner'] == data['Player_2']].groupby("Pts_2").size().reset_index(
        name="Wins").sort_values(
        by='Pts_2').rename(columns={"Pts_2": "Pts"})
    points_wins = pd.concat([points_wins_1, points_wins_2])
    points_wins = points_wins.groupby("Pts")['Wins'].sum().reset_index().sort_values(by='Pts')
    print(points_wins)

    plt.figure(figsize=(12, 6))
    # plt.xscale('log')
    sns.histplot(x='Pts', data=points_wins, color='blue')
    # Dodanie tytułu i etykiet osi
    plt.title('Liczba wygranych meczów w zależności od pozycji w rankingu', fontsize=16)
    plt.xlabel('Liczba punktów w rankingu', fontsize=12)
    plt.ylabel('Liczba wygranych meczów', fontsize=12)
    # Wyświetlenie wykresu
    # plt.tight_layout()
    plt.show()


def rank_wins_turnament_plot(conn):
    # tutaj zmieniamy tylko serie turnieju do filtra
    data = pd.read_sql_query("SELECT * FROM matches where Series = 'ATP250' and Rank_1 < 129 and Rank_2 < 129", conn)
    ranking_wins_1 = data[data['Winner'] == data['Player_1']].groupby("Rank_1").size().reset_index(
        name="Wins").sort_values(
        by='Rank_1').rename(columns={"Rank_1": "Rank"})
    ranking_wins_2 = data[data['Winner'] == data['Player_2']].groupby("Rank_2").size().reset_index(
        name="Wins").sort_values(
        by='Rank_2').rename(columns={"Rank_2": "Rank"})
    ranking_wins = pd.concat([ranking_wins_1, ranking_wins_2])
    ranking_wins = ranking_wins.groupby("Rank")['Wins'].sum().reset_index().sort_values(by='Rank')

    matches_played_1 = data.groupby("Rank_1").size().reset_index(name='Matches_played').rename(
        columns={"Rank_1": "Rank"})
    matches_played_2 = data.groupby("Rank_2").size().reset_index(name='Matches_played').rename(
        columns={"Rank_2": "Rank"})
    matches_played = pd.concat([matches_played_1, matches_played_2])
    matches_played = matches_played.groupby('Rank')['Matches_played'].sum().reset_index().sort_values(by='Rank')

    win_percentage = pd.merge(matches_played, ranking_wins, on="Rank", how='left')
    win_percentage['Win_Percentage'] = (win_percentage["Wins"] / win_percentage['Matches_played']) * 100

    print(win_percentage)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Rank', y='Win_Percentage', data=win_percentage, color='blue')
    # Dodanie tytułu i etykiet osi
    plt.title('Procent wygranych meczów w zależności od pozycji w rankingu (ATP250)', fontsize=16)
    plt.xlabel('Pozycja w rankingu', fontsize=12)
    plt.ylabel('Procent wygranych meczów', fontsize=12)
    # Ograniczenie widocznych wartości osi X (np. do top 50 pozycji)
    plt.xticks(rotation=90)
    # plt.xlim(0, 50)  # Wyświetlanie tylko pierwszych 50 pozycji w rankingu
    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()


def ranking_difference_percentage(data):
    data["Ranking_Difference"] = abs(data["Rank_1"] - data["Rank_2"])
    all_matches = data.groupby("Ranking_Difference").size().reset_index(name="All")

    low_win_1 = data[(data["Winner"] == data["Player_1"]) & (data["Rank_1"] > data["Rank_2"])]
    low_win_2 = data[(data["Winner"] == data["Player_2"]) & (data["Rank_1"] < data["Rank_2"])]
    low_win = pd.concat([low_win_1, low_win_2])

    low_win = low_win.groupby("Ranking_Difference").size().reset_index(name="Wins")

    win_percentage = pd.merge(all_matches, low_win, on="Ranking_Difference", how="left")
    win_percentage["win_percentage"] = (win_percentage["Wins"] / win_percentage["All"]) * 100
    win_percentage = win_percentage.dropna()

    print(win_percentage.to_string())

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Ranking_Difference', y='win_percentage', data=win_percentage, color='blue')
    # Dodanie tytułu i etykiet osi
    plt.title('Procent wygranych meczów przez zawodnika o niższym rankingu', fontsize=16)
    plt.xlabel('Różnica rankingów', fontsize=12)
    plt.ylabel('Procent wygranych meczów', fontsize=12)
    # Ograniczenie widocznych wartości osi X (np. do top 50 pozycji)
    plt.xticks(rotation=90)
    plt.xlim(0, 100)  # Wyświetlanie tylko pierwszych 50 pozycji w rankingu
    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()


def prematch_odds_accuracy(data):
    data["odds_difference"] = abs(data['Odd_1'] - data['Odd_2'])

    data["odds_difference_bins"] = pd.cut(data['odds_difference'], bins=30)

    all_matches = data.groupby('odds_difference_bins').size().reset_index(name="All")
    low_win_1 = data[(data['Winner'] == data['Player_1']) & (data['Odd_1'] < data['Odd_2'])]
    low_win_2 = data[(data['Winner'] == data['Player_2']) & (data['Odd_1'] > data['Odd_2'])]
    low_win = pd.concat([low_win_1, low_win_2])
    low_win = low_win.groupby("odds_difference_bins").size().reset_index(name="Wins")

    accuracy = pd.merge(all_matches, low_win, on="odds_difference_bins", how="left")
    accuracy["percentage"] = (accuracy['Wins']/accuracy['All'])*100

    print(accuracy.to_string())

    plt.figure(figsize=(12, 6))
    sns.barplot(x='odds_difference_bins', y="percentage", data=accuracy, color='blue')
    # Dodanie tytułu i etykiet osi
    plt.title('Dokładność obstawiania w zależności od różnic kursów', fontsize=16)
    plt.xlabel('Przedziały różnic kursów (odd_1 - odd_2)', fontsize=12)
    plt.ylabel('Dokładność obstawiania (%)', fontsize=12)

    # Rotacja etykiet na osi X
    plt.xticks(rotation=45)

    # wyświetlamy tylko kilka pierwszych, kolejne prawie wszystkie mają 100%(z drobnymi pomijalnymi wyjątkami)
    plt.xlim(0, 10)
    # Wyświetlenie wykresu
    plt.tight_layout()
    plt.show()


def random_forest():
    data['Rank_Difference'] = abs(data['Rank_1'] - data['Rank_2'])
    data['Odds_Difference'] = abs(data['Odd_1'] - data['Odd_2'])
    data['Winner_Label'] = (data['Winner'] == data['Player_1']).astype(int)
    data.dropna()
    X = data[['Rank_Difference', 'Rank_1', 'Rank_2', 'Odd_1', 'Odd_2', 'Odds_Difference']]
    y = data['Winner_Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Ważność cech
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    # Wykres ważności cech
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
    plt.title('Ważność cech w modelu Random Forest')
    plt.xlabel('Ważność')
    plt.ylabel('Cecha')
    plt.tight_layout()
    plt.show()


def test1(data):
    # Test dla różnicy rankingów
    rank_diff_winner = data[data['Winner_Label'] == 1]['Rank_Difference']
    rank_diff_loser = data[data['Winner_Label'] == 0]['Rank_Difference']

    t_stat_rank, p_value_rank = ttest_ind(rank_diff_winner, rank_diff_loser)
    print(f"Rank Difference: T-statistic = {t_stat_rank:.2f}, P-value = {p_value_rank:.4f}")

    # Test dla różnicy kursów
    odds_diff_winner = data[data['Winner_Label'] == 1]['Odds_Difference']
    odds_diff_loser = data[data['Winner_Label'] == 0]['Odds_Difference']

    t_stat_odds, p_value_odds = ttest_ind(odds_diff_winner, odds_diff_loser)
    print(f"Odds Difference: T-statistic = {t_stat_odds:.2f}, P-value = {p_value_odds:.4f}")


def chi_kwadrat(data):
    # Tworzenie kategorii dla różnicy kursów
    data['Odds_Bin'] = pd.cut(data['Odds_Difference'], bins=[0, 2, 4, 6, 8, 10, float('inf')],
                              labels=['0-2', '2-4', '4-6', '6-8', '8-10', '>10'])

    # Tworzenie tabeli kontyngencji
    contingency_table = pd.crosstab(data['Odds_Bin'], data['Winner_Label'])

    # Test chi-kwadrat
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-squared: {chi2:.2f}, P-value: {p:.4f}")


def ranking_bins_win_percentage(data):
    # Dodanie kolumny z kategoriami pozycji w rankingu
    data['Rank_Bin'] = pd.cut(data['Rank_1'], bins=[0, 10, 20, 50, 100, float('inf')],
                              labels=['Top 10', 'Top 20', 'Top 50', 'Top 100', '>100'])

    # Policzenie procentu wygranych dla każdej kategorii
    rank_win_percentage = data.groupby('Rank_Bin')['Winner_Label'].mean() * 100

    print(rank_win_percentage)

    # Wizualizacja wyników
    rank_win_percentage.plot(kind='bar', color='blue', figsize=(10, 6),
                             title='Procent wygranych zawodników według pozycji w rankingu')
    plt.xlabel('Pozycja w rankingu')
    plt.ylabel('Procent wygranych (%)')
    plt.tight_layout()
    plt.show()


def regresja(data):
    # Dodanie stałej do modelu (wymagane przez statsmodels)
    X = sm.add_constant(data[['Rank_Difference', 'Rank_1', 'Rank_2', 'Odds_Difference', 'Odd_1', 'Odd_2']])
    y = data['Winner_Label']

    # Regresja liniowa
    model = sm.Logit(y, X).fit()
    print(model.summary())

    coefficients = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.params
    }).sort_values(by='Coefficient', ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients, palette='viridis')
    plt.title('Współczynniki regresji logistycznej')
    plt.xlabel('Współczynnik')
    plt.ylabel('Cecha')
    plt.tight_layout()
    plt.show()


def weryfikacja_hipotez(data):
    # Grupujemy turnieje według rangi
    def categorize_tournament(row):
        if "Grand Slam" in row['Series']:
            return "Grand Slam"
        elif "Masters 1000" in row['Series']:
            return "Masters 1000"
        elif "ATP500" in row['Series']:
            return "ATP500"
        elif "ATP250" in row['Series']:
            return "ATP250"
        else:
            return "Other"

    # Dodanie kolumny kategorii turnieju
    data['Tournament_Category'] = data.apply(categorize_tournament)

    # Podział zawodników na grupy według rankingu (np. Top 10, 11-50, 51-100, 101+)
    def rank_group(rank):
        if rank <= 10:
            return "Top 10"
        elif rank <= 50:
            return "11-50"
        elif rank <= 100:
            return "51-100"
        else:
            return "101+"

    # Dodanie kolumny z grupą rankingu
    data['Rank_Group'] = data['Rank_1'].apply(rank_group)

    # Obliczenie statystyk dla liczby rozegranych meczów
    matches_played = data.groupby(['Rank_Group', 'Tournament_Category']).size().reset_index(name='Matches_Played')

    # Obliczenie procentu wygranych
    data['Win_1'] = (data['Winner_Label'] == 1).astype(int)
    win_stats = data.groupby(['Rank_Group', 'Tournament_Category'])['Win_1'].mean().reset_index(name='Win_Percentage')

    # Łączenie danych
    stats = pd.merge(matches_played, win_stats, on=['Rank_Group', 'Tournament_Category'])

    # Wizualizacja: liczba rozegranych meczów
    plt.figure(figsize=(12, 6))
    sns.barplot(data=matches_played, x='Tournament_Category', y='Matches_Played', hue='Rank_Group')
    plt.title('Liczba rozegranych meczów w różnych kategoriach turniejów')
    plt.xlabel('Kategoria turnieju')
    plt.ylabel('Liczba meczów')
    plt.legend(title='Grupa rankingu')
    plt.show()

    # Wizualizacja: procent wygranych
    plt.figure(figsize=(12, 6))
    sns.barplot(data=win_stats, x='Tournament_Category', y='Win_Percentage', hue='Rank_Group')
    plt.title('Procent wygranych w różnych kategoriach turniejów')
    plt.xlabel('Kategoria turnieju')
    plt.ylabel('Procent wygranych')
    plt.legend(title='Grupa rankingu')
    plt.show()

    # Wyświetlenie tabeli z wynikami
    print(stats)


# csv_to_sql()


conn = sqlite3.connect('tennis_data.db')
data = pd.read_sql_query("SELECT * FROM matches", conn)
conn.close()
# print(data.describe().to_string())

# rank_wins_plot(data)

# rankPoints_wins_plot(data)

# rank_wins_turnament_plot(conn)

# ranking_difference_percentage(data)

# prematch_odds_accuracy(data)

# ranking_bins_win_percentage(data)


# MODEL STATYSTYCZNY:

data['Rank_Difference'] = abs(data['Rank_1'] - data['Rank_2'])
data['Odds_Difference'] = abs(data['Odd_1'] - data['Odd_2'])
data['Winner_Label'] = (data['Winner'] == data['Player_1']).astype(int)

# random_forest()

# test1(data)

# chi_kwadrat(data)

# regresja(data)

weryfikacja_hipotez(data)