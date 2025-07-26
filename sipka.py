def get_legal_positions(self):
        actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        # akcije sahovskoh konja
        knight_actions = [(1, 2), (1, -2), (-1, 2), (-1, -2), (2, 1), (2, -1), (-2, 1), (-2, -1)]
        diagonal_actions = [(1,1), (1,-1), (-1,1), (-1,-1)]
        rook_actions = []
        bishop_actions = []
        queen_actions = []


        #za topa:
        for i in range(self.board.cols):
            rook_actions.append((0, i))
            rook_actions.append((0, -i))
        
        for i in range(self.board.rows):
            rook_actions.append((i, 0))
            rook_actions.append((-i, 0))

        #za lovca
        for i in range(1, self.board.rows):
            for j in range(1, self.board.cols):
                if i == j and (i, j) not in bishop_actions:
                    bishop_actions.append((i, j))
                    bishop_actions.append((i, -j))
                    bishop_actions.append((-i, j))
                    bishop_actions.append((-i, -j))

        #za kraljicu:
        for r in rook_actions:
            queen_actions.append(r)
        for b in bishop_actions:
            queen_actions.append(b)

        row, col = self.position 
        new_positions = []

        if self.teleport == True:
            new_positions.append((self.teleports[0], "Teleport"))
            return new_positions
        
        if self.checkpoint_count == 1:
            actions = queen_actions
        elif self.checkpoint_count >= 2 :
            actions = rook_actions
            
        if actions == rook_actions:          
            for d_row, d_col in actions: 
                new_row = row + d_row
                new_col = col + d_col
                if not self.board.is_out_of_bounds(new_row, new_col) and not self.board.hits_wall(new_row, new_col):
                    hit_wall = False
                    start, end = min(row, new_row), max(row, new_row)
                    for i in range(start, end):
                        if self.board.hits_wall(i, new_col):
                            hit_wall = True

                    start, end = min(col, new_col), max(col, new_col)
                    for i in range(start, end):
                        if self.board.hits_wall(new_row, i):
                            hit_wall = True                            
                    
                    if not hit_wall:
                        new_positions.append(((new_row, new_col), (d_row, d_col)))

        elif actions == bishop_actions: 
            for d_row, d_col in actions:
                new_row = row + d_row
                new_col = col + d_col
                if not self.board.is_out_of_bounds(new_row, new_col) and not self.board.hits_wall(new_row, new_col):
                    hit_wall = False

                    if d_row < 0:
                        i = -1
                    else:
                        i = 1
                    current_row = row + i
                    
                    if d_col < 0:
                        j = -1
                    else:
                        j = 1
                    current_col = col + j

                    while current_row != new_row and current_col != new_col:
                        if self.board.hits_wall(current_row, current_col):
                            hit_wall = True
                            break
                        current_row += i
                        current_col += j
                    
                    if not hit_wall:
                        new_positions.append(((new_row, new_col), (d_row, d_col)))
        
        elif actions == queen_actions:
            for d_row, d_col in actions: 
                new_row = row + d_row
                new_col = col + d_col
                if not self.board.is_out_of_bounds(new_row, new_col) and not self.board.hits_wall(new_row, new_col):
                    hit_wall = False

                    if d_row == 0 or d_col == 0:
                        if d_col == 0:
                            start, end = min(row, new_row), max(row, new_row)
                            for i in range(start+1, end):
                                if self.board.hits_wall(i, new_col):
                                    hit_wall = True
                        
                        elif d_row == 0:
                            start, end = min(col, new_col), max(col, new_col)
                            for i in range(start+1, end):
                                if self.board.hits_wall(new_row, i):
                                    hit_wall = True
                    
                    elif abs(d_row) == abs(d_col):
                        if d_row < 0:
                            i = -1
                        else:
                            i = 1
                        current_row = row + i
                        
                        if d_col < 0:
                            j = -1
                        else:
                            j = 1
                        current_col = col + j

                        while current_row != new_row and current_col != new_col:
                            if self.board.hits_wall(current_row, current_col):
                                hit_wall = True
                                break
                            current_row += i
                            current_col += j

                    if not hit_wall:
                        new_positions.append(((new_row, new_col), (d_row, d_col)))
            
        else:
            for d_row, d_col in actions: 
                new_row = row + d_row
                new_col = col + d_col 

                if not self.board.is_out_of_bounds(new_row, new_col) and not self.board.hits_wall(new_row, new_col):
                    new_positions.append(((new_row, new_col), (d_row, d_col)))

        return new_positions
