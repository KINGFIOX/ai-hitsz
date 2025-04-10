# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def uniSearch(init : callable, update : callable, end : callable) -> list:
    info = init()
    problem : SearchProblem = info["problem"]
    frontier = info["frontier"]
    while not frontier.isEmpty():
        state = frontier.peek()[0] # just peek
        if problem.isGoalState(state):
            return end(info) # would pop
        update(info) # would pop
    return []


def depthFirstSearch(problem: SearchProblem) -> list:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    def init():
        reached = set()
        frontier = util.Stack() # (neighbors, actions in path)
        frontier.push((problem.getStartState(), [], 0))
        return { "reached" : reached, "frontier" : frontier, "problem" : problem }

    def update(info):
        frontier = info["frontier"]
        reached = info["reached"]
        state, path, path_cost = frontier.pop()
        if not state in reached:
            reached.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                newPath = path + [action]
                frontier.push((successor, newPath, path_cost + stepCost))

    def end(info):
        frontier = info["frontier"]
        _, path, _ = frontier.pop()
        return path

    return uniSearch(init, update, end)

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    def init():
        reached = set()
        frontier = util.Queue() # (neighbors, actions in path)
        frontier.push((problem.getStartState(), [], 0))
        return { "reached" : reached, "frontier" : frontier, "problem" : problem }

    def update(info):
        reached = info["reached"]
        frontier = info["frontier"]
        state, path, path_cost = frontier.pop()
        if not state in reached:
            reached.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                newPath = path + [action]
                frontier.push((successor, newPath, path_cost + stepCost))

    def end(info):
        frontier = info["frontier"]
        _, path, _ = frontier.pop()
        return path
        
    return uniSearch(init, update, end)


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    def init():
        reached = set()
        frontier = util.PriorityQueue() # (state, path, cost)
        frontier.push((problem.getStartState(), [], 0), 0)
        return { "reached" : reached, "frontier" : frontier, "problem" : problem }

    def update(info):
        reached = info["reached"]
        frontier = info["frontier"]
        state, path, path_cost = frontier.pop()
        if not state in reached:
            reached.add(state)
            for successor, action, stepCost in problem.getSuccessors(state):
                newPath = path + [action]
                new_cost = path_cost + stepCost
                frontier.push((successor, newPath, new_cost), new_cost)

    def end(info):
        frontier = info["frontier"]
        _, path, _ = frontier.pop()
        return path 

    return uniSearch(init, update, end)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def manhattanHeuristic(state, problem):
    """
    Calculate the Manhattan distance between the current state and the goal state.
    """
    x, y = state
    goal_x, goal_y = problem.getGoalState()
    return abs(x - goal_x) + abs(y - goal_y)

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    start = problem.getStartState()
    frontier = util.PriorityQueue()
    frontier.push((start, ()), 0)
    came_from = { start : None }
    cost_so_far = { start : 0 }
    cur_succ = { start : None } # action: cur -> succ

    while not frontier.isEmpty():
        cur, _ = frontier.pop()       
        if problem.isGoalState(cur):
            path = []
            while cur != start:
                path.append(cur_succ[cur])
                cur = came_from[cur]
            path.reverse()
            return path
        for succ, action, stepCost in problem.getSuccessors(cur):
            new_cost = cost_so_far[cur] + stepCost
            if succ not in cost_so_far or new_cost < cost_so_far[succ]: # update cost and frontier
                cost_so_far[succ] = new_cost
                priority = new_cost + heuristic(succ, problem)
                frontier.push((succ, ()), priority)
                came_from[succ] = cur # record the path
                cur_succ[succ] = action
    
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
