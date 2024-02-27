



@misc{poke_env,
    author       = {Haris Sahovic},
    title        = {Poke-env: pokemon AI in python},
    url          = {https://github.com/hsahovic/poke-env}
}

# PokeBot-Zero
AI for Pokemon Showdown using RL methods inspired by AlphaZero/AlphaGo projects

## Introduction
This project is an AI implementation for Pokémon battles using the poke-env library and a Pokémon Showdown server. It's designed to experiment with and develop AI strategies in the Pokémon game environment.

Getting Started
Prerequisites
Python >= 3.8
Pokémon Showdown server
Setting Up the Environment
Clone the repository:

bash
Copy code
git clone <https://github.com/chenbm4/PokeBot-Zero.git>
cd <PokeBot-Zero>
Initialize Submodules:

bash
Copy code
git submodule update --init --recursive
Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install dependencies:

```
pip install -r requirements.txt
## For development:
pip install -r requirements-dev.txt
Setting Up Pokémon Showdown
Clone and setup the Pokémon Showdown server:
```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security

Credits
poke-env
This project uses the poke-env library for interacting with Pokémon Showdown. poke-env is developed by Haris Sahovic.

Reference: Sahovic, H. (n.d.). Poke-env: pokemon AI in python. Retrieved from https://github.com/hsahovic/poke-env
Pokémon Showdown
Owner:

Guangcong Luo [Zarel] - Development, Design, Sysadmin
Staff:
Andrew Werner [HoeenHero] - Development
Annika L. [Annika] - Development
Chris Monsanto [chaos] - Development, Sysadmin
Kris Johnson [Kris] - Development
Leonard Craft III [DaWoblefet] - Research (game mechanics)
Mathieu Dias-Martins [Marty-D] - Research (game mechanics), Development
Mia A [Mia] - Development
Contributors:

Full list of contributors can be found at http://pokemonshowdown.com/credits
License
Include information about your project's license here.