# Pac-Man
A Python implementation of artificial intelligence search algorithms to solve problems within the Berkeley Pac-Man environment. The [Pac-Man Projects, developed at UC Berkeley](http://ai.berkeley.edu), apply AI concepts to the classic arcade game. I help Pac-Man find food, avoid ghosts, and maximise his game score using uninformed and informed state-space search, probabilistic inference, and reinforcement learning.

This project uses Python 2.7.13 plus NumPy 1.13.1 and SciPy 0.19.1.

## How to Play
Use WASD or arrow keys to control Pac-Man. To start an interactive game, type at the command line:

~~~~
python pacman.py
~~~~

<p align="center">
<img src="https://github.com/thiadeliria/Pacman/blob/master/gifs/interactive.gif" width="540" />
</p>

To see how Pac-Man fares using search algorithms, we can define some variables:
~~~~
python pacman.py -l MAZE_TYPE -p SearchAgent -a fn=SEARCH_ALGO
~~~~
where `MAZE_TYPE` defines the map layout, and `SearchAgent` navigates Pac-Man through the maze according to the algorithm supplied in the `SEARCH_ALGO` parameter.


## Multiagent Search

### Minimax
Now you will write an adversarial search agent in the provided MinimaxAgent class stub in multiAgents.py. Your minimax agent should work with any number of ghosts, so you’ll have to write an algorithm that is slightly more general than what you’ve previously seen in lecture. In particular, your minimax tree will have multiple min layers (one for each ghost) for every max layer.

We run
~~~~
python autograder.py -q q2
~~~~

##Hints and Observations
* Hint: Implement the algorithm recursively using helper function(s).
* The correct implementation of minimax will lead to Pacman losing the game in some tests. This is not a problem: as it is correct behaviour, it will pass the tests.
* The evaluation function for the Pacman test in this part is already written (self.evaluationFunction). You shouldn’t change this function, but recognize that now we’re evaluating states rather than actions, as we were for the reflex agent. Look-ahead agents evaluate future states whereas reflex agents evaluate actions from the current state.
* The minimax values of the initial state in the minimaxClassic layout are 9, 8, 7, -492 for depths 1, 2, 3 and 4 respectively. Note that your minimax agent will often win (665/1000 games for us) despite the dire prediction of depth 4 minimax.
~~~~
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
~~~~

* Pacman is always agent 0, and the agents move in order of increasing agent index.
* All states in minimax should be GameStates, either passed in to getAction or generated via GameState.generateSuccessor. In this project, you will not be abstracting to simplified states.
* On larger boards such as openClassic and mediumClassic (the default), you’ll find Pacman to be good at not dying, but quite bad at winning. He’ll often thrash around without making progress. He might even thrash around right next to a dot without eating it because he doesn’t know where he’d go after eating that dot. Don’t worry if you see this behavior, question 5 will clean up all of these issues.
When Pacman believes that his death is unavoidable, he will try to end the game as soon as possible because of the constant penalty for living. Sometimes, this is the wrong thing to do with random ghosts, but minimax agents always assume the worst:
~~~~
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
~~~~





### Alpha-Beta Pruning
Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree, in AlphaBetaAgent. Again, your algorithm will be slightly more general than the pseudocode from lecture, so part of the challenge is to extend the alpha-beta pruning logic appropriately to multiple minimizer agents.

You should see a speed-up (perhaps depth 3 alpha-beta will run as fast as depth 2 minimax). Ideally, depth 3 on smallClassic should run in just a few seconds per move or faster.
~~~~
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
~~~~

The pseudo-code below represents the algorithm you should implement for this question.
![alt text](https://inst.eecs.berkeley.edu/~cs188/sp20/assets/images/alpha_beta_impl.png)

We run
~~~~
python autograder.py -q q3
~~~~


### Expectimax
Minimax and alpha-beta are great, but they both assume that you are playing against an adversary who makes optimal decisions. As anyone who has ever won tic-tac-toe can tell you, this is not always the case. In this question you will implement the ExpectimaxAgent, which is useful for modeling probabilistic behavior of agents who may make suboptimal choices.

As with the search and constraint satisfaction problems covered so far in this class, the beauty of these algorithms is their general applicability. To expedite your own development, we’ve supplied some test cases based on generic trees. You can debug your implementation on small the game trees using the command:

~~~~
python autograder.py -q q4
~~~~

Once your algorithm is working on small trees, you can observe its success in Pacman. Random ghosts are of course not optimal minimax agents, and so modeling them with minimax search may not be appropriate. ExpectimaxAgent, will no longer take the min over all ghost actions, but the expectation according to your agent’s model of how the ghosts act. To simplify your code, assume you will only be running against an adversary which chooses amongst their getLegalActions uniformly at random.

To see how the ExpectimaxAgent behaves in Pacman, run:
~~~~
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
~~~~
