import search
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

""" EVALUATION FUNCTION """
def evaluate_state(state, problem):
    """
    Heuristic / scoring function for state
    """
    grid = state["board"]
    heights = get_height_list(grid)
    #return -(10*get_num_holes(grid) + 2**(get_height(heights)) + bumpiness(heights) + average_height(heights))
    return -1.0/1000*(72*get_height(heights) + 75*average_height(heights) + 442*get_num_holes(grid) + 56*bumpiness(heights) + 352 * valleys(grid, heights))


class TetrisSearchProblem(search.SearchProblem):
    def __init__(self, lookahead=1,verbose=False):
        # Number of pieces for which we want to look ahead for
        # `1` meaning we look only at the next piece
        # `2` meaning we base it off the next two pieces, and so on
        self.lookahead = lookahead
        self.verbose = verbose

        # Generate random sequence of pieces for offline tetris
        NUM_PIECES = 10
        self.all_pieces = [random.choice(tetris.SHAPES) for i in xrange(NUM_PIECES)]

        if demo:
            self.all_pieces = [tetris.LINE_SHAPE, tetris.SQUARE_SHAPE, tetris.SQUARE_SHAPE,
                 tetris.T_SHAPE, tetris.Z_SHAPE, tetris.L_SHAPE, tetris.LINE_SHAPE,
                 tetris.T_SHAPE, tetris.T_SHAPE] + \
                [tetris.LINE_SHAPE, tetris.SQUARE_SHAPE, tetris.INVERT_L_SHAPE, tetris.LINE_SHAPE, tetris.LINE_SHAPE] + \
                [random.choice(tetris.SHAPES) for i in xrange(30)] 

        # Taken from tetris.py: initial board setup
        self.initial_board = []  
        for i in range(GRID_HEIGHT):
            self.initial_board.append([])
            for j in range(GRID_WIDTH):
                self.initial_board[i].append(None)   

    def getStartState(self):
        # Tuple of configuration and past grids
        return { "pieces": self.all_pieces, "board": self.initial_board }

    def isGoalState(self, state):
        # Should be a goal state when there are no pieces left
        return len(state["pieces"]) == 0

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


    def getReward(self, state, action):
        """
        Returns the empirical reward for a state and action

        Args:
            state: the current state
            action: a piece (and maybe an x-coordinate)?

        Returns: a numeric value
        
        TODO
        """
        pass

    def getSuccessors(self, state):
        """
        Return a list of successor nodes
        using the board and current piece. 
        """
        if len(state["pieces"]) == 0:
            return None
        
        new_piece_type = state["pieces"][0]
        grid = state["board"]

        successors = []

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
            while can_move_right:
                # Copying the grids here might explode memory, but I think keeping
                # a reference to the same grid repeatedly is going to be really dangerous
                piece_copy = copy.deepcopy(rotated_piece)
                grid_copy = copy.deepcopy(grid)

                # Move the piece all the way down
                while piece_copy.move_down(grid_copy): pass

                # Add the block to the grid and clear lines
                # filter out a successor that makes a game ending move
                try:
                    merge_grid_block(grid_copy, piece_copy)

                    # UNCOMMENT FOR AN INFINITE GAME
                    #  push a new random piece to replace the one we played
                    # piece = random.choice(tetris.SHAPES)
                    # state["pieces"].append(piece)

                    successors.append({
                        "board": grid_copy,
                        "pieces": state["pieces"][1:] 
                    })
                except:
                    pass
                    #print "failed move!"

                # Try the next configuration
                can_move_right = rotated_piece.move_right(grid)  # has side-effects

        return successors

    def getCostOfActions(self, actions):
        pass

def test_tetris(ntrial=10, lookahead=1, heuristic=evaluate_state, watchGames=False, verbose=False):
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
    for i in range(ntrial):
        problem = TetrisSearchProblem(lookahead=lookahead,verbose=verbose)

        current_node = None
        
        # Game loop: keep playing the game until all of the pieces are done
        while current_node is None or len(current_node["pieces"]) > 0:
            print current_node
            game_replay, goal_node = search.aStarSearch(problem, heuristic)
            current_node = goal_node

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

