from tensors import boardToTensor, moveToTensor
from dataset import ChessPositionDataset
import chess.pgn
import torch
import numpy

pgn = open("dataset\lichess_elite_2020-06.pgn")
game = chess.pgn.read_game(pgn)

positions = []
moves = []
board = game.board()

for move in game.mainline_moves():
    boardArray = boardToTensor(board)
    moveArray = moveToTensor(move.uci())

    

    positions.append(boardArray)
    moves.append(moveArray)
    board.push(move)

boardTensor = torch.FloatTensor(numpy.array(positions))
moveTensor = torch.FloatTensor(numpy.array(moves))


dataset = ChessPositionDataset(boardTensor, moveTensor)
print(len(dataset))

pgn.close()
#test
#d
#new