import chess


material = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:100}
def calculateMaterial(board, isWhite): #more negative means bot has more material
    if board.is_checkmate():
        botColor = chess.BLACK if isWhite else chess.WHITE
        if board.turn == botColor:
            return 10000
        else:
            return -10000
    materialDiff = 0
    for i in range(64):
        currentPiece = board.piece_at(i)
        if currentPiece is None:
            continue
        

        if currentPiece.color == chess.WHITE:
            materialDiff += material[currentPiece.piece_type]
        else:
            materialDiff -= material[currentPiece.piece_type]
    if not isWhite:  
        return materialDiff
    return -materialDiff



def filterBlunders(board, possibleMoves, isWhite):
    depth = 4
    if not possibleMoves:
        return
    
    materialBefore = calculateMaterial(board, isWhite)
    bestMove = None
    bestEval = float('-inf')
    
    for move in possibleMoves:
        board.push(move)        
        moveEval = minimax(board, depth - 1, float('-inf'), float('inf'), False, isWhite)
        board.pop()

        if moveEval > bestEval:
            bestEval = moveEval
            bestMove = move      
        
    if bestMove == None:
        return possibleMoves[0]
    return bestMove


def minimax(board, depth, alpha, beta, botTurn, isWhite):
    if depth == 0 or board.is_game_over():
        return calculateMaterial(board, isWhite)
    
    if botTurn:
        maxEval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            evaluate = minimax(board, depth - 1, alpha, beta, False, isWhite)
            board.pop()


            maxEval = max(maxEval, evaluate)
            alpha = max(alpha, evaluate)

            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            evaluate = minimax(board, depth - 1, alpha, beta, True, isWhite)
            board.pop()
            
            minEval = min(minEval, evaluate)
            beta = min(beta, evaluate)
            if beta <= alpha:
                break  

        return minEval
