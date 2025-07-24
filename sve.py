class RobotState(State):

    def __init__(self, board: Board, parent: State=None, position: tuple=None, goal_position: tuple=None, action: tuple=None):
        super().__init__(board, parent, position, goal_position, action)

        if self.parent is None:
            self.cost = 0
            self.num_boxes = 0
        else:
            self.cost = self.parent.cost + 1
            self.num_boxes = self.parent.num_boxes
            
            #Ako menja smer => cena mu je tri puta veca
            if self.action != self.parent.action:
                self.cost = self.cost + 2

            #Pokupi samo 3 kutije
            if self.num_boxes < 3:
                #Sve plave kutije
                self.checkpoints = list(self.checkpoints)

                for c in self.checkpoints:
                    if self.position == c:
                        self.checkpoints.remove(c)
                        #Za 3 kutije
                        self.num_boxes += 1
                        break

                self.checkpoints = tuple(self.checkpoints)

            #Teleport
            if self.position in self.teleports:
                self.teleport = True

                self.teleports = list(self.teleports)
                for t in self.teleports:
                    if self.position == t:
                        self.teleports.remove(t)
                        break
                self.teleports = tuple(self.teleports)
                
                if self.parent.teleport == True:
                    self.teleport = False

            #Vatra
            self.cost += 100/(self.euclidian_distance(self.position, self.fire) + 1)**2

    def get_legal_positions(self):
        
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        both = [(0, 1), (0, -1), (1, 0), (-1, 0),(1, 1), (-1, -1), (1, -1), (-1,1)] #i dijagonalno i normalno
        knight_actions = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)] #akcije sahovskoh konja

        row, col = self.position 
        new_positions = []

        #Teleport
        if self.teleport == True:
            new_positions.append((self.teleports[0], "Teleport"))
            return new_positions
        
        #Kad pokupi sve moze I dijagonalno
        if len(self.checkpoints) == 0:
            actions = both

        for d_row, d_col in actions: 
            new_row = row + d_row
            new_col = col + d_col 

            if not self.board.is_out_of_bounds(new_row, new_col) and not self.board.hits_wall(new_row, new_col):
                new_positions.append(((new_row, new_col), (d_row, d_col)))
        return new_positions

    def is_final_state(self):
        return self.position == self.goal_position and self.num_boxes == 3

    def unique_hash(self):
        return str(self.position) + str(self.num_boxes)
    
    def get_cost_estimate(self):
        return self.manhattan_distance(self.position, self.goal_position)
        
    def get_current_cost(self):
        return self.cost
    
    def manhattan_distance(self, pointA, pointB):
        return abs(pointA[0] - pointB[0]) + abs(pointA[1] - pointB[1])
    
    def euclidian_distance(self, pointA, pointB):
        return math.sqrt((pointA[0] - pointB[0])**2 + (pointA[1] - pointB[1])**2)
    
    def diagonal_distance(self, pointA, pointB):
        return max([abs(pointA[0] - pointB[0]), abs(pointA[1] - pointB[1])])