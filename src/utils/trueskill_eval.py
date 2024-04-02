import json
import trueskill

class RLSkillEvaluator:
    def __init__(self, draw_probability=0.02, mu=25, sigma=25/3, beta=25/6, tau=0.0):
        self.env = trueskill.TrueSkill(draw_probability=draw_probability, mu=mu, 
                                       sigma=sigma, beta=beta, tau=tau)
        self.players = {}

    def add_player(self, player_id):
        if player_id not in self.players:
            self.players[player_id] = self.env.create_rating()
    
    def update_skills(self, winner_id, loser_id):
        if winner_id in self.players and loser_id in self.players:
            winner_rating, loser_rating = self.players[winner_id], self.players[loser_id]
            updated_winner, updated_loser = self.env.rate_1vs1(winner_rating, loser_rating)
            self.players[winner_id], self.players[loser_id] = updated_winner, updated_loser

    def get_player_skill(self, player_id):
        return self.players.get(player_id, None)

    def compare_players(self, player1_id, player2_id):
        if player1_id in self.players and player2_id in self.players:
            delta = self.players[player1_id].mu - self.players[player2_id].mu
            return delta
        return None
    
    def save_leaderboard(self, filepath):
        # Convert player ratings to a savable format
        savable_leaderboard = {player_id: {'mu': rating.mu, 'sigma': rating.sigma}
                               for player_id, rating in self.players.items()}
        with open(filepath, 'w') as file:
            json.dump(savable_leaderboard, file, indent=4)

    def load_leaderboard(self, filepath):
        with open(filepath, 'r') as file:
            loaded_leaderboard = json.load(file)
        for player_id, ratings in loaded_leaderboard.items():
            self.players[player_id] = self.env.create_rating(mu=ratings['mu'], sigma=ratings['sigma'])