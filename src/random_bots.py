import asyncio
import time
import random

from poke_env.player import Player

class TestPlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            move = random.choice(battle.available_moves)
            return self.create_order(move)
        
        # If no attack is available, a random switch will be made
        elif battle.available_switches:
            switch = random.choice(battle.available_switches)
            return self.create_order(switch)
        
        # no moves or switches available
        else:
            print("No moves or switches available")
            return self.choose_default_move(battle)
    

async def main():
    start = time.time()

    # Create two players
    player1 = TestPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)
    player2 = TestPlayer(battle_format="gen8randombattle", max_concurrent_battles=0)

    await player1.battle_against(player2, n_battles=100)

    print(
        "Player 1 won %d / 100 battles [%.2f seconds]"
        % (
            player1.n_won_battles,
            time.time() - start
        )
    )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
        
