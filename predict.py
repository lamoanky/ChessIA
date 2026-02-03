import chess
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as functional
from tensors import Tensors
from neuralNetwork import ChessModel, device
from blunder import filterBlunders
class Predict:
    def __init__(self, tensors):
        self.model = ChessModel()
        self.path = "model17v3.pth"        
        self.model.load_state_dict(torch.load(self.path, device))
        self.model.to(device)
        self.model.eval()
        self.tensors = tensors

    def valueToMove(self, value):
        fromSquare = chess.square_name(value//64)
        toSquare = chess.square_name(value%64)

        return fromSquare + toSquare
        

    def getTop5(self, moveList, board, isWhite):
        possibleMoves = []
        for i in moveList:
            
            uciMove = self.valueToMove(i)
            if uciMove[:2] == uciMove[2:4]:
                continue
            pMove = uciMove + "q"
            try:
                move = chess.Move.from_uci(uciMove)
                promoteMove = chess.Move.from_uci(pMove)
                if promoteMove in board.legal_moves:
                    possibleMoves.insert(0, promoteMove)
                elif move in board.legal_moves:
                    possibleMoves.append(move)
            except chess.InvalidMoveError:
                continue
        
        return filterBlunders(board, possibleMoves, isWhite)
            
    def predictMove(self, board, isWhite):
        position = self.tensors.boardToTensor(board)
        boardTensor = torch.FloatTensor(numpy.array(position)).unsqueeze(0).to(device)
        
        
        with torch.no_grad():
            output = self.model(boardTensor)
            move = self.valueToMove(output.argmax().item())
            values, indices = torch.topk(output, 4096)

            moveList = indices.tolist()[0]

            return self.getTop5(moveList, board, isWhite)