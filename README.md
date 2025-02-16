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

#### current challenge
the Qlearning agent gets stuck in a loop.
what I have done to improve that?
1. use a menhattan distance for wighting rewards.
2. implement a gerridy row search, to reduce the memory usage.

#### what search ideas I have implemented (A* and gerridy algorithms):
1. a* search
2. a* row masked search
3. a* fully masked search

challengs on row search:
for the even n, (like 4*4 puzzles), there might be some parity issue, and that algorithms does not garrenty to solve the problem.
what is parrity issue? check this out:

```python
# Solving row for puzzle:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [15 14  0 13]]
Target goal:
[[ 1,  2,  3,  4],
[ 5,  6,  7,  8],
[ 9, 10, 11, 12],
[13, 14, 15,  0]]
```

this algorithm, consider each row fixed when that row is becaume sorted; so on the lastes row, we may face the unsolvable problem. But again, this only happens if we consider the each row fixed, so by solving the 2 last rows toghether, this parrity issue becaume solved.