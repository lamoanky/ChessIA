import chess
import numpy

board = chess.Board()

def boardToTensor(board):
    tensor = numpy.zeros((13,8,8))

    for square in range(64):
        currentPiece = board.piece_at(square)
        if currentPiece == None: 
            continue
        
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        pieceType = currentPiece.piece_type - 1 
        """
        PAWN : 1
        KNIGHT : 2
        BISHOP : 3
        ROOK : 4
        QUEEN : 5
        KING : 6
        """
        if currentPiece.color == chess.WHITE:
            tensor[pieceType, 7-rank, file] = 1
        else:
            tensor[pieceType + 6, 7-rank, file] = 1

    for move in board.legal_moves:
        toSquare = move.to_square
        rank = chess.square_rank(toSquare)
        file = chess.square_file(toSquare)
        tensor[12, 7-rank, file] = 1

    return tensor
        
    
print(boardToTensor(board))

def moveToValue(UCImove): #changing this so that it returns a value instead of a 2x8x8 tensor
    move = chess.Move.from_uci(UCImove)
    fromSquare = move.from_square
    toSquare = move.to_square

    return fromSquare * 64 + toSquare
    

"""
move = chess.Move.from_uci(UCImove)
tensor = numpy.zeros((2,8,8))
fromSquare = move.from_square
toSquare = move.to_square

fromRank = chess.square_rank(fromSquare)
fromFile = chess.square_file(fromSquare)

toRank = chess.square_rank(toSquare)
toFile = chess.square_file(toSquare)

tensor[0, 7 - fromRank, fromFile] = 1
tensor[1, 7 - toRank, toFile] = 1

return tensor
"""