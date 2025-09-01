# Bat Algorithm for game playing Agent

## Project Overview

This repository contains the implementation of an intelligent agent that plays a Space Invaders–like game using a handcrafted neural network whose weights are optimized via the Bat Algorithm (metaheuristic). The focus is on running within the constraints of the assignment:
- Population size ≤ 100 bats
- Algorithm iterations ≤ 1000
- Maximum runtime: 12 hours

## Repository Structure

- `game/`: core game logic and simulation environment
- `agents/`: neural network and agent implementation
- `metaheuristics/`: Bat Algorithm optimizer (refined version)
- `training/`: scripts to train the agent and run experiments
- `evaluation/`: scripts for statistical analysis and result visualization
- `play_trained_agent.py`: script to run the trained agent visually
- `generate_bat_agent_results.py`: runs multiple evaluations to produce scores for statistical testing
- `best_agent_weights.npy`: saved weights of the best-trained agent
- `bat_agent_best_result.txt`: best results from batch runs

Dependencies:

Python 3

Standard libraries only: numpy, matplotlib, scipy, seaborn (already included in the repo or standard Python installations).

## How to Run

1. **Train the agent:**

   ```bash
   python training/train_agent.py

  This will run the Bat Algorithm optimization, output the best score, and save:

best_agent_weights.npy

fitness_evolution.png

Total runtime printed in the terminal

2. **Play with the trained agent:**

python play_trained_agent.py

Watch the agent play in real time and see the final score.

3. **Generate statistics for comparison (e.g., versus humans or rule-based agents):**

First generate multiple runs:
```bash
  python generate_bat_agent_results.py
```
This saves a result file (bat_agent_result.txt) with 30 individual scores.

Then generate analysis plots and statistical test results:

```bash
python evaluation/evaluate_results.py

