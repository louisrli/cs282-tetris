import random

import tetris
import copy
import collections
import numpy as np
import sys
from time import sleep
import string
import matplotlib.pyplot as plt
debug = True
demo = True
TESTMODE = True

GRID_HEIGHT = 20
GRID_WIDTH = 6

def print_grid(grid, block=None):
    """
    Print ASCII version of our tetris for debugging
    """
    for y in xrange(GRID_HEIGHT):
        if debug: 
            # Column numbers
            print "%2d" % y,

        for x in xrange(GRID_WIDTH):
            block = grid[y][x]
            if block:
                print block,
            else:
                print ".",
        print  # Newline at the end of a row
    print

def clear_lines(grid):
  """
  Clear lines from a grid. Mutates grid.
  
  Taken from tetris.py, Tetris.clear_lines()
  
  Returns:
      The number of lines cleared
  """
  count=0
  for i in range(GRID_HEIGHT):
      full=True
      for j in range(GRID_WIDTH):
          if(grid[i][j] is None): 
              full=False
              break
      if(full):
          count+=1
          for j in range(GRID_WIDTH):
              grid[i][j]=None
  i=GRID_HEIGHT-1
  j=GRID_HEIGHT-2
  while(i>0 and j>=0):
      null=True
      for k in range(GRID_WIDTH):
          if(grid[i][k] is not None):
              null=False
              break
      if(null):
          j=min(i-1,j)
          while(j>=0 and null):
              null=True
              for k in range(GRID_WIDTH):
                  if(grid[j][k] is not None):
                      null=False
                      break
              if(null): j-=1
          if(j<0): break
          for k in range(GRID_WIDTH):
              grid[i][k]=grid[j][k]
              grid[j][k]=None
              if(grid[i][k] is not None): grid[i][k].y=tetris.HALF_WIDTH+i*tetris.FULL_WIDTH
          j-=1
      i-=1
  
  return count


def merge_grid_block(grid, block):
    """
    Given a grid and a block, add the block to the grid.
    This is called to "merge" a block into a grid. You can
    think of the grid as the static part of pieces that have already
    been placed down. `block` is the current piece that you're looking to 
    place down.

    See settle_block() from tetris.py. Same thing, except without game logic

    Returns:
        Nothing, modifies grid via side-effects
    """
    for square in block.squares:
        y, x = square.y / tetris.FULL_WIDTH, square.x / tetris.FULL_WIDTH
        if grid[y][x]:
            #print get_height(grid)
            raise Exception("Tried to put a Tetris block where another piece already exists")
        else:
            grid[y][x]=square
    return

""" CALL THIS BEFORE GET HEIGHT OR AVERAGE HEIGHT OR ETC """
def get_height_list(grid):
    heights = []
    for index in range(GRID_WIDTH):
        temp = [i for i, x in enumerate([col[index] for col in grid][::-1]) if x != None]
        heights.append(0 if len(temp) == 0 else max(temp)+1) # 0-indexed lol  
    return heights

def get_height(heights):
    """
    Given a list of heights, calculates the maximum height of any column
    Returns:
        int representing the maximum height of any column
    """
    return max(heights)

def average_height(heights):
    return float(sum(heights)) / GRID_WIDTH


"""bumpiness: sum of absolute height differences between neighboring columns"""
def bumpiness(heights):
    bumpiness = [heights[i+1]-heights[i] for i in range(0, GRID_WIDTH-1)]
    return sum([abs(b) for b in bumpiness])

def valleys(grid, heights):
    valleys = 0
    for i in range(2, GRID_HEIGHT):
        for j in range(GRID_WIDTH):
            if grid[i][j] is None and grid[i-1][j] is None and grid[i-2][j] is None:
                if j == 0:
                    if heights[j+1] >= heights[j] + 3:
                        valleys += 1

                elif j == GRID_WIDTH-1:
                    if heights[j-1] >= heights[j] + 3:
                        valleys += 1
                else:
                    if heights[j-1] >= heights[j] + 3 and heights[j+1] >= heights[j] + 3:
                        valleys += 1
    return valleys

def get_num_holes(g):
    """
    Given a grid, calculates the number of ''holes'' in the placed pieces
    Returns:
        int - number of holes
    """
    grid = copy.deepcopy(g)
    # use sets to avoid dups
    holes = set()
    # first for loop finds initial underhangs
    for i in range(len(grid) - 1, 0, -1): # row
        for j in range(len(grid[i])): # col
            if i - 1 >= 0 and grid[i][j] is None and grid[i-1][j] is not None:
                holes.add((i, j))
    # new copy because can't change set while iterating.
    all_holes = copy.deepcopy(holes)
    # for each find earlier keep digging down to see how many holes additionally there are
    for i, j in holes:
        while i + 1 < len(grid) and grid[i + 1][j] is None:
            all_holes.add((i + 1, j))
            i += 1
    return len(all_holes)

def get_lines_cleared(gnew, gold):
    diff_lines = get_height(gnew) - get_height(gold)
    if diff_lines > -4:
        return -100 # strongly prefer clearing 4 at a time
    else:
        return 0
    return

def get_ttl(grid):
    """
    Returns the top two levels of the grid
    """
    height = max(get_height_list(grid))
    top_index = GRID_HEIGHT - height

    # Take the top two rows if we're almost dead
    if height == GRID_HEIGHT or height == GRID_HEIGHT - 1:
        top_index = 0

    rows = grid[top_index:top_index + 2]
    as_ones = []
    for row in rows:
        as_ones.append(tuple([0 if s is None else 1 for s in row]))
    return tuple(as_ones)

def convert_state(state, hole='high',k=0,num_next=1):
    """
    Converts a state from the internal representation

    Args:
        hole: 
            'high' -- represent holes by the height of highest holes
            'count' -- represent holes by number of holes per column
        k: The height of the skyline to examine (top k rows)
        num_next: The number of next pieces to look at
    """
    return get_ttl(state['board'])

def ignore():
    skyline = get_height_list(state['board'])
    holes = []
    for col in range(GRID_WIDTH):
        if hole == 'count':
            col_holes = sum([1 for x in range(GRID_HEIGHT-1,GRID_HEIGHT-1-skyline[col],-1) if not state['board'][x][col]])
            holes.append(col_holes)
        elif hole == 'high':
            highest_hole = -1
            for i in range(col-1,-1,-1):
                if not state['board'][i][col]:
                    highest_hole = i
                    continue
            holes.append(highest_hole)
        else: 
            raise Exception
    converted = []
    converted.append(tuple([max(0,col-k) for k in skyline]))
    converted.append(tuple(holes))
    converted.append(tuple(state['pieces'][:num_next]))
    return tuple(converted)


class TetrisAgent():
    def __init__(self, epsilon, alpha, gamma=0.95):
        self.gamma = gamma
        self.epsilon_func = epsilon
        self.alpha_func = alpha
        self.iteration = 1
        self.value_table = {}
        self.last_state = None
        self.last_action = None
        self.reset()

    def reset(self):
        self.value_table = {}
        self.last_state = None
        self.last_action = None
        self.iteration = 1

    def interact(self, reward, next_state, problem):
        # Handle start of episode
        actions = problem.get_possible_actions()
        random.shuffle(actions)
        if reward is None:
            self.last_state = next_state
            self.last_action = random.choice(actions)
            return self.last_action
        if not self.value_table.get(next_state):
            self.value_table[next_state] = collections.defaultdict(float)
        if not self.value_table.get(self.last_state):
            self.value_table[self.last_state] = collections.defaultdict(float)
        q_vals = [self.value_table[next_state][action] for action in actions]
        max_action = actions[np.argmax(q_vals)]
        delta = reward + self.gamma*self.value_table[next_state][max_action] - self.value_table[self.last_state][self.last_action]

        self.value_table[self.last_state][self.last_action] += self.alpha_func(self.iteration)*delta
        self.iteration += 1
        
        self.last_state = next_state
        if random.random() < self.epsilon_func(self.iteration):
            self.last_action = random.choice(actions)
        else:
            self.last_action = max_action
        
        return self.last_action

    def _get_num_keys(self):
        """ 
        For debugging. Return the number of keys in value_table
        """
        keys = 0
        for k in self.value_table.keys():
            keys += len(self.value_table[k].keys())
        return keys



class TetrisLearningProblem():
    def __init__(self, gamma=0.98, verbose=False):
        self.verbose = verbose

        # Initialized by reset()
        self.gameover = None
        self.board = None
        self.pieces = None

        # Set up the board and pieces
        self.reset()

    def reset(self):
        """
        Resets the tetris board to empty and re-initializes with a random set of pieces
        """
        self.gameover = False

        # Generate random sequence of pieces for offline tetris
        NUM_PIECES = 1000
        self.pieces = [random.choice(tetris.SHAPES) for i in xrange(NUM_PIECES)]

        # Set up an empty board
        self.board = []  
        for i in range(GRID_HEIGHT):
            self.board.append([])
            for j in range(GRID_WIDTH):
                self.board[i].append(None)   

    def _get_internal_state(self):
        """
        Return an *internal* representation of the state as the board and the list of pieces
        remaining.
        """
        return { "board": self.board, "pieces": self.pieces }

    def is_terminal(self):
        """
        Returns true if the game is over: either out of pieces or lost on board
        """
        return len(self.pieces) == 0 or self.gameover

    def perform_action(self, action):
        """
        Perform an action. An action is just a Block().
        (basically just changes the current board to whatever preview_action gives)

        This should be the only way to mutate the internal state.
        """
        LOSS_REWARD = -100
        new_board = self.preview_action(action)
        if new_board is None:
            self.gameover = True
            return LOSS_REWARD, self._get_internal_state()

        # Compute the number of lines cleared
        lines_cleared = clear_lines(new_board)  # Side effecting
        assert(lines_cleared <= 4)

        # Subtract the number of holes
        num_holes = get_num_holes(new_board)

        line_clear_reward = 0 if (lines_cleared == 0) else 2**(lines_cleared - 1)
        reward = line_clear_reward - num_holes
        #reward = evaluate_state(self._get_internal_state())

        # Update internal state
        self.board = new_board
        self.pieces = self.pieces[1:]
        return reward, self._get_internal_state()

    def preview_action(self, action):
        """
        Given an action, return the new board, but does not internally modify the current
        state of the world

        Returns:
            A board
            None if placing the piece down leads to a loss
        """
        grid = copy.deepcopy(self.board)
        piece = copy.deepcopy(action)

        # Move the piece all the way down on the current board
        while piece.move_down(grid): pass

        # Add the block to the grid and clear lines
        try: 
            merge_grid_block(grid, piece)
        except:
            return None
        return grid
   
    def get_possible_actions(self):
        """
        Based on the current state, compute the list of possible actions
        (memoized since this is probably expensive)

        For the tetris domain, this is going to be:
            (every possible rotation) x (every possible x-position)

        As such, we encode an action as a Block() class from the framework.
        Block contains stateful information about a piece's rotation and x-axis.
        """
        if len(self.pieces) == 0:
            return []

        # Put the piece in the right place
        new_piece_type = self.pieces[0]
        grid = copy.deepcopy(self.board)

        possible_actions = []

        new_piece = tetris.Block(new_piece_type)

        # Because we're leveraging tetris.py, we have a lot of 
        # side-effecting code going on -- have to be careful
        possible_rotations = self._generateRotations(new_piece, grid)

        # Starting from the left-hand side this moves the 
        # piece to the right one column (i.e. every horizontal position).
        # Then we move the piece all the way down.
        # In this way, we enumerate all possible subsequent configurations.
        for rotated_piece in possible_rotations:
            can_move_right = True
            offset = 0  # Distance from the left
            while can_move_right:
                piece_snapshot = copy.deepcopy(rotated_piece)
                possible_actions.append(piece_snapshot)
                can_move_right = rotated_piece.move_right(grid)  # has side-effects
                offset += 1

        return possible_actions


    def _generateRotations(self, piece, grid):
        """
        Args:
            piece: Block() object
        Returns:
            List of Block objects for the possible rotations
        """
        rotated_pieces = []
        TOTAL_ROTATIONS = 4  # 0, 90, 180, 270
        for num_cw_rotations in xrange(TOTAL_ROTATIONS): 
            # Make a copy of the piece so we can manipulate it
            new_piece = copy.deepcopy(piece)

            # Short circuit logic for rotating the correct number of times CW
            # This might be buggy...not really sure what his can_CW function checks for
            did_rotate = True
            for _ in xrange(num_cw_rotations):
                if new_piece.can_CW(grid):
                    new_piece.rotate_CW(grid)
                else:
                    did_rotate = False

            # By default, tetris.py instantiates pieces in the middle.
            # Move it all the way to the left. move_left() side-effects.
            while new_piece.move_left(grid): pass

            if not did_rotate:
                continue
            else:
                rotated_pieces.append(new_piece)

        return rotated_pieces


def test_tetris(ntrials=1, nepisodes=1000, niter=100):
    """
    Test harness
    """
    problem = TetrisLearningProblem()
    agent = TetrisAgent(epsilon=lambda x: .01, alpha=lambda x: pow(.99,x))
    reward_mat = np.zeros((ntrials,nepisodes))
    for n in range(ntrials):
        total_rewards = []
        problem.reset()
        agent.reset()
        for e in range(nepisodes):
            rewards = []
            problem.reset()
            state = convert_state(problem._get_internal_state(), k=17)
            reward = None
            for i in range(niter):
                if problem.is_terminal():
                    break
                #sleep(.5)
                #print_grid(problem._get_internal_state()['board'])
                action = agent.interact(reward, state, problem)
                reward, state = problem.perform_action(action)
                state = convert_state(state, k=17)
                rewards.append(reward)
            reward_mat[n][e] = sum(rewards)
        fig = plt.figure()
        plt.plot(np.sum(reward_mat,axis=0)/ntrials)
        plt.xlabel('episodes')
        plt.ylabel('cumulative rewards')
        plt.title('cumulative rewards vs episodes')
        plt.show()

def main():
    test_tetris()


if __name__ == '__main__':
    main()

