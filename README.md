# Tic-Tac-Toe-AI-with-Minimax

Tic-tac-toe (American English), noughts and crosses (Commonwealth English), or Xs and Os/“X’y O’sies” (Ireland), is a paper-and-pencil game for two players, X and O, who take turns marking the spaces in a 3×3 grid. The player who succeeds in placing three of their marks in a diagonal, horizontal, or vertical row is the winner. It is a solved game with a forced draw assuming best play from both players.

### Gameplay
In order to win the game, a player must place three of their marks in a horizontal, vertical, or diagonal row.

The following example game is won by the first player, X:
<img src ="https://en.wikipedia.org/wiki/File:Tic-tac-toe-game-1.svg"/>

Players soon discover that the best play from both parties leads to a draw. Hence, tic-tac-toe is most often played by young children, who often have not yet discovered the optimal strategy.
Because of the simplicity of tic-tac-toe, it is often used as a pedagogical tool for teaching the concepts of good sportsmanship and the branch of artificial intelligence that deals with the searching of game trees. It is straightforward to write a computer program to play tic-tac-toe perfectly or to enumerate the 765 essentially different positions (the state space complexity) or the 26,830 possible games up to rotations and reflections (the game tree complexity) on this space.[1] If played optimally by both players, the game always ends in a draw, making tic-tac-toe a futile game.[2]

The game can be generalized to an m,n,k-game in which two players alternate placing stones of their own color on an m×n board, with the goal of getting k of their own color in a row. Tic-tac-toe is the (3,3,3)-game.[3] Harary's generalized tic-tac-toe is an even broader generalization of tic-tac-toe. It can also be generalized as a nd game. Tic-tac-toe is the game where n equals 3 and d equals 2.[4] It can be generalised even further by playing on an arbitrary incidence structure, where rows are lines and cells are points. Tic-tac-toe is the game given by the incidence structure shown to the right, consisting of nine points, three horizontal lines, three vertical lines, and two diagonal lines, each line consisting of at least three points.

<img src = "Screen Shot 2021-01-05 at 3.57.57 AM.png"/>

### Minimax

Minimax is a decision rule used in artificial intelligence, decision theory, game theory, statistics, and philosophy for minimizing the possible loss for a worst case (maximum loss) scenario. When dealing with gains, it is referred to as "maximin"—to maximize the minimum gain. Originally formulated for n-player zero-sum game theory, covering both the cases where players take alternate moves and those where they make simultaneous moves, it has also been extended to more complex games and to general decision-making in the presence of uncertainty.

<h5>Minimax</h5> is a kind of backtracking algorithm that is used in decision making and game theory to find the optimal move for a player, assuming that your opponent also plays optimally. It is widely used in two player turn-based games such as Tic-Tac-Toe, Backgammon, Mancala, Chess, etc.

In Minimax the two players are called maximizer and minimizer. The maximizer tries to get the highest score possible while the minimizer tries to do the opposite and get the lowest score possible.

Every board state has a value associated with it. In a given state if the maximizer has upper hand then, the score of the board will tend to be some positive value. If the minimizer has the upper hand in that board state then it will tend to be some negative value. The values of the board are calculated by some heuristics which are unique for every type of game.

 #### Which move you would make as a maximizing player considering that your opponent also plays optimally?
 
Since this is a backtracking based algorithm, it tries all possible moves, then backtracks and makes a decision.

Maximizer goes LEFT: It is now the minimizers turn. The minimizer now has a choice between 3 and 5. Being the minimizer it will definitely choose the least among both, that is 3
Maximizer goes RIGHT: It is now the minimizers turn. The minimizer now has a choice between 2 and 9. He will choose 2 as it is the least among the two values.
Being the maximizer you would choose the larger value that is 3. Hence the optimal move for the maximizer is to go LEFT and the optimal value is 3.

<img src = "https://media.geeksforgeeks.org/wp-content/uploads/TIC_TAC.jpg"/>

This image depicts all the possible paths that the game can take from the root board state. It is often called the Game Tree. 
The 3 possible scenarios in the above example are : 
 

Left Move : If X plays [2,0]. Then O will play [2,1] and win the game. The value of this move is -10
Middle Move : If X plays [2,1]. Then O will play [2,2] which draws the game. The value of this move is 0
Right Move : If X plays [2,2]. Then he will win the game. The value of this move is +10;
Remember, even though X has a possibility of winning if he plays the middle move, O will never let that happen and will choose to draw instead.
Therefore the best choice for X, is to play [2,2], which will guarantee a victory for him.
We do encourage our readers to try giving various inputs and understanding why the AI chose to play that move. Minimax may confuse programmers as it it thinks several moves in advance and is very hard to debug at times. Remember this implementation of minimax algorithm can be applied any 2 player board game with some minor changes to the board structure and how we iterate through the moves. Also sometimes it is impossible for minimax to compute every possible game state for complex games like Chess. Hence we only compute upto a certain depth and use the evaluation function to calculate the value of the board.
Stay tuned for next weeks article where we shall be discussing about Alpha-Beta pruning that can drastically improve the time taken by minimax to traverse a game tree.


As seen in the above article, each leaf node had a value associated with it. We had stored this value in an array. But in the real world when we are creating a program to play Tic-Tac-Toe, Chess, Backgamon, etc. we need to implement a function that calculates the value of the board depending on the placement of pieces on the board. This function is often known as Evaluation Function. It is sometimes also called Heuristic Function.

The evaluation function is unique for every type of game. In this post, evaluation function for the game Tic-Tac-Toe is discussed. The basic idea behind the evaluation function is to give a high value for a board if maximizer‘s turn or a low value for the board if minimizer‘s turn.

For this scenario let us consider X as the maximizer and O as the minimizer.

Let us build our evaluation function :

If X wins on the board we give it a positive value of +10.

<img src = "https://media.geeksforgeeks.org/wp-content/uploads/TicTacToe.png"/>

If O wins on the board we give it a negative value of -10.

<img src = "https://media.geeksforgeeks.org/wp-content/uploads/TicTacToe1.png"/>

If no one has won or the game results in a draw then we give a value of +0.

<img src = "https://media.geeksforgeeks.org/wp-content/uploads/TicTacToe2-1.png"/>

We could have chosen any positive / negative value other than 10. For the sake of simplicity we chose 10 for the sake of simplicity we shall use lower case ‘x’ and lower case ‘o’ to represent the players and an underscore ‘_’ to represent a blank space on the board.

If we represent our board as a 3×3 2D character matrix, like char board[3][3]; then we have to check each row, each column and the diagonals to check if either of the players have gotten 3 in a row.

Let us combine what we have learnt so far about minimax and evaluation function to write a proper Tic-Tac-Toe AI (Artificial Intelligence) that plays a perfect game. This AI will consider all possible scenarios and makes the most optimal move.
 

Finding the Best Move :
We shall be introducing a new function called findBestMove(). This function evaluates all the available moves using minimax() and then returns the best move the maximizer can make. The pseudocode is as follows : 

```
function findBestMove(board):
    bestMove = NULL
    for each move in board :
        if current move is better than bestMove
            bestMove = current move
    return bestMove
```

##### Minimax :
To check whether or not the current move is better than the best move we take the help of minimax() function which will consider all the possible ways the game can go and returns the best value for that move, assuming the opponent also plays optimally 
The code for the maximizer and minimizer in the minimax() function is similar to findBestMove() , the only difference is, instead of returning a move, it will return a value. Here is the pseudocode : 
function minimax(board, depth, isMaximizingPlayer):
```
    if current board state is a terminal state :
        return value of the board
    
    if isMaximizingPlayer :
        bestVal = -INFINITY 
        for each move in board :
            value = minimax(board, depth+1, false)
            bestVal = max( bestVal, value) 
        return bestVal

    else :
        bestVal = +INFINITY 
        for each move in board :
            value = minimax(board, depth+1, true)
            bestVal = min( bestVal, value) 
        return bestVal
```


Checking for GameOver state :
To check whether the game is over and to make sure there are no moves left we use isMovesLeft() function. It is a simple straightforward function which checks whether a move is available or not and returns true or false respectively. Pseudocode is as follows :
 
```
function isMovesLeft(board):
    for each cell in board:
        if current cell is empty:
            return true
    return false
``` 

Making our AI smarter :
One final step is to make our AI a little bit smarter. Even though the following AI plays perfectly, it might choose to make a move which will result in a slower victory or a faster loss. Lets take an example and explain it.
Assume that there are 2 possible ways for X to win the game from a give board state.
 

Move A : X can win in 2 move
Move B : X can win in 4 moves
Our evaluation function will return a value of +10 for both moves A and B. Even though the move A is better because it ensures a faster victory, our AI may choose B sometimes. To overcome this problem we subtract the depth value from the evaluated score. This means that in case of a victory it will choose a the victory which takes least number of moves and in case of a loss it will try to prolong the game and play as many moves as possible. So the new evaluated value will be
 

Move A will have a value of +10 – 2 = 8
Move B will have a value of +10 – 4 = 6
Now since move A has a higher score compared to move B our AI will choose move A over move B. The same thing must be applied to the minimizer. Instead of subtracting the depth we add the depth value as the minimizer always tries to get, as negative a value as possible. We can subtract the depth either inside the evaluation function or outside it. Anywhere is fine. I have chosen to do it outside the function. Pseudocode implementation is as follows.
 
```
if maximizer has won:
    return WIN_SCORE – depth

else if minimizer has won:
    return LOOSE_SCORE + depth
```


### References

https:www.geeksforgeeks.org
https://en.wikipedia.org/wiki/Minimax
https://www.youtube.com/user/shiffman


