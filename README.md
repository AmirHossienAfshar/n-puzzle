# n-puzzle
The (n²-1) sliding puzzle with Python and Reinforcement Learning.

![Sliding Puzzle AI](assets/prototype.jpg)


### how to run
1. Clone the repository:
```
git clone https://github.com/AmirHossienAfshar/n-puzzle.git
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run the program:
```
python main.py
```

### current challenge
the Qlearning agent gets stuck in a loop.
what I have done to improve that?
1. use a menhattan distance for wighting rewards.