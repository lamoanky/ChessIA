import chess
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tensors import boardToTensor
from neuralNetwork import ChessModel, device
from blunder import filterBlunders
model = ChessModel()
def loadModel():
    path = "model6v2.pth"

    
    model.load_state_dict(torch.load(path, device))
    model.to(device)
    model.eval()

def valueToMove(value):
    fromSquare = chess.square_name(value//64)
    toSquare = chess.square_name(value%64)

    return fromSquare + toSquare
    

def getTop5(moveList, board, isWhite):
    possibleMoves = []
    for i in moveList:
        
        uciMove = valueToMove(i)
        if uciMove[:2] == uciMove[2:4]:
            continue
        pMove = uciMove + "q"
        try:
            move = chess.Move.from_uci(uciMove)
            promoteMove = chess.Move.from_uci(pMove)
            if promoteMove in board.legal_moves:
                return promoteMove
            if move in board.legal_moves:
                possibleMoves.append(move)
        except chess.InvalidMoveError:
            continue
    
    return filterBlunders(board, possibleMoves, isWhite)
        
def predictMove(board, isWhite):
    loadModel()
    position = boardToTensor(board)
    boardTensor = torch.FloatTensor(numpy.array(position)).unsqueeze(0).to(device)
    
    
    with torch.no_grad():
        output = model(boardTensor)
        move = valueToMove(output.argmax().item())
        values, indices = torch.topk(output, 4096)

        moveList = indices.tolist()[0]

        return getTop5(moveList, board, isWhite)
