import random
import tetris
import copy
import sys
from time import sleep
import string

debug = False
demo = True
TESTMODE = True

GRID_HEIGHT = 20
GRID_WIDTH = 10

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

class TetrisLearningProblem():
    def __init__(self, gamma=0.95, verbose=False):
        self.verbose = verbose
        self.gamma = 0.95
        self.epsilon_func = None  # TODO
        self.alpha_func =  None   # TODO

        # Initialized by reset()
        self.board = None
        self.pieces = None

        # Set up the board and pieces
        self.reset()

    def reset(self):
        """
        Resets the tetris board to empty and re-initializes with a random set of pieces
        """
        # Generate random sequence of pieces for offline tetris
        NUM_PIECES = 10
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

    def observe(self):
        """
        Return a representation of the state for the agent to use
        """
        pass

    def is_terminal(self):
        """
        Returns true if the game is over: either out of pieces or lost on board
        """
        # TODO(louisli): or if the game is lost
        return len(self.pieces) == 0

    def perform_action(self, action):
        """
        Perform an action. An action is just a Block().
        (basically just changes the current board to whatever preview_action gives)
        """
        new_board = self.preview_action(action)

        # Compute the number of lines cleared
        lines_cleared = 1

        assert(lines_cleared <= 4)

        # Subtract the number of holes
        num_holes = get_num_holes(new_board)

        reward = 2**lines_cleared - num_holes

        self.board = new_board
        reward = self._get_reward()
        return reward

    def preview_action(self, action):
        """
        Given an action, return the new board, but does not internally modify the current
        state of the world

        Returns:
            A board
        """
        grid = copy.deepcopy(self.board)
        piece = copy.deepcopy(action)

        # Move the piece all the way down on the current board
        while piece.move_down(grid): pass

        # Add the block to the grid and clear lines
        try: 
            merge_grid_block(grid, piece)
        except:
            raise Exception
        return grid
   
    def get_possible_actions(self):
        """
        Based on the current state, return the list of possible actions

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

    def _convert_state(state, hole='high',k=0,num_next=1):
        """
        Converts a state from the internal representation

        Args:
            hole: 
                'high' -- represent holes by the height of highest holes
                'count' -- represent holes by number of holes per column
            k: The height of the skyline to examine (top k rows)
            num_next: The number of next pieces to look at
        """
        skyline = get_height_list(state['board'])
        holes = []
        for col in skyline:
            if hole == 'count':
                col_holes = sum([1 for x in range(col) if not state['board'][x][col]])
                holes.append(col_holes)
            elif hole == 'high':
                highest_hole = -1
                for i in range(col,-1,-1):
                    if not state['board'][i][col]:
                        highest_hole = i
                        continue
                holes.append(highest_hole)
            else: 
                raise Exception
            
        converted = {}
        converted['skyline'] = [max(0,col-k) for k in skyline]
        converted['holes'] = holes
        converted['next'] = state['pieces'][:num_next]
        return converted

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


    def _get_reward(self, state, action):
        """
        Returns the reward for being in the current state
        Normally, reward is r(s, a), but in our case, it only depends on the current state.
        We'll return this value when the agent performs an action.

        Arg:
            The current state (after performing an action)
        Returns: 
            a numeric value
        """

        return 1  # TODO

def test_tetris(ntrial=10, lookahead=1, heuristic=None, watchGames=False, verbose=False):
    """
    Test harness
    """

    if lookahead < 1:
        print "Bad Lookahead! Please pick 1 for no lookahead, 2 for 1-piece, etc..."
        return
    else:
        print "Lookahead: " + str(lookahead - 1) + " pieces"
    if verbose:
        print "Verbose Printing Enabled"
    else:
        print "Verbose Printing Disabled"
    if watchGames:
        print "Game Replay Enabled"
    else:
        print "Game Replay Disabled"

    total_lines = []
    for n in range(ntrial):
        problem = TetrisLearningProblem(lookahead=lookahead,verbose=verbose)
        value_table = {}
        for e in range(nepisode):
            last_state = problem.convertState(problem.getStartState())
            next_state = last_state
            last_successor = None
            for i in range(niter):
                successors = problem.getSuccessors(last_state)
                if not value_table.get(next_state):
                    value_table[next_state] = collections.defaultdict(float)
                q_vals = [value_table[next_state][successor] for successor in successors]
                max_successor = succesors[np.argmax(q_vals)]
                delta = problem.getReward() + problem.gamma*value_table[next_state][max_succesor] - value_table[last_state][last_successor]
                value_table[last_state][last_successor] += problem.alpha_func(i)*delta
                last_state = next_state
                if random.random() < problem.epsilon_func(i):
                    last_successor = random.choice(tetris.SHAPES)
                else:
                    last_action = max_successor

                # perform action



            print current_node
            game_replay, goal_node = None, None
 
            if watchGames:
                for grid in game_replay:
                    print_grid(grid)
                    sleep(0.2)
                sleep(2)

            lines_cleared = 0
            for j in range(len(game_replay)-1):
                before = max(get_height_list(game_replay[j]))
                after = max(get_height_list(game_replay[j+1]))
                if after < before:
                    lines_cleared += before - after

            print "Lines cleared: " + str(lines_cleared)

            with open('gameLogs/trial_3'+str(i)+'_linesCleared='+str(lines_cleared)+'.txt', 'w') as fout:
                for g in game_replay:
                    fout.write(str(g))
                    fout.write('\n')
            break
            #return # TODO: remove once we have a real goal state

        total_lines.append(lines_cleared)

    print "Lines by Game: " + str(total_lines)
    print "Total Lines: " + str(sum(total_lines)) + " in " + str(ntrial) + " games."

def stringify_board(board):
    """
    Takes the board as a printed list and returns it as a pretty string.

    Returns:
        A string
    """
    parsed = string.replace(str(board), ',', '')
    parsed = string.replace(parsed, 'None', '.')
    parsed = string.lstrip(parsed, '[[')
    parsed = string.rstrip(parsed, ']]\n')
    
    parselist = string.split(parsed, '] [')
    return '\n'.join(parselist)


def watchReplay(filename):
    with open(filename) as f:
        for line in f:
            parsed = string.replace(line, ',', '')
            parsed = string.replace(parsed, 'None', '.')
            parsed = string.lstrip(parsed, '[[')
            parsed = string.rstrip(parsed, ']]\n')
            
            parselist = string.split(parsed, '] [')
            for p in parselist:
                print p
            sleep(0.5)

def printHelp():
    print "Usage: python algo.py [OPTIONS]"
    print "\t-h, --help\tPrints this help dialog"
    print "\t-t, --tetris\tRuns the tetris AI simulation"
    print "\t\t ARGS: [# trials] [lookahead = 1,2,...] [watch replay=0,1] [verbose=0,1]"
    print "\t-r, --replay\tWatch a game replay"
    print "\t\t ARGS: [gamelog]"
    # print "\t-d, --demo\tWatch the class demo"

def main():
    return  # TODO: remove

    if len(sys.argv) < 2:
        printHelp()
        return

    # HELP
    if len(sys.argv) == 2 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        printHelp()
        return

    # REPLAY MODE
    if len(sys.argv) == 3 and (sys.argv[1] == "-r" or sys.argv[1] == "--replay"):
        watchReplay(sys.argv[2])

    # AI SIMULATION
    if len(sys.argv) == 6 and (sys.argv[1] == "-t" or sys.argv[1] == "--tetris"):
        test_tetris(ntrial=int(sys.argv[2]), lookahead=int(sys.argv[3]), watchGames=int(sys.argv[4]), verbose=int(sys.argv[5]))


if __name__ == '__main__':
    main()

