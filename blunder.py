import chess
material = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}
def calculateMaterial(board, isWhite):
    materialDiff = 0
    for i in range(64):
        currentPiece = board.piece_at(i)
        if currentPiece is None:
            continue
        if currentPiece.piece_type == chess.KING:
            continue
        

        if currentPiece.color == chess.WHITE:
            materialDiff += material[currentPiece.piece_type]
        else:
            materialDiff -= material[currentPiece.piece_type]
    if isWhite:  
        return materialDiff
    return -materialDiff



def filterBlunders(board, possibleMoves, isWhite):
    materialBefore = calculateMaterial(board, isWhite)
    bestMove = possibleMoves[0]

    for AImove in possibleMoves:
        board.push(AImove)        
        worstCase = materialBefore 

        for humanMove in board.legal_moves:
            board.push(humanMove)
            materialAfter = calculateMaterial(board, isWhite)
            board.pop()

            if materialAfter > worstCase:
                worstCase = materialAfter
    
        board.pop()
    

        if worstCase <= materialBefore:
            return AImove
        
    return bestMove