from tensors import boardToTensor, moveToValue
from dataset import ChessPositionDataset
import chess.pgn
import torch
import numpy

pgn = open("dataset/lichess_elite_2020-06.pgn")


positions = []
moves = []
currentGame = 0
maxGames = 100
print("Initializing game...")

for i in range(maxGames):
    game = chess.pgn.read_game(pgn)
    board = game.board()
    for move in game.mainline_moves():
        boardArray = boardToTensor(board)
        moveValue = moveToValue(move.uci())

        

        positions.append(boardArray)
        moves.append(moveValue)
        board.push(move)

boardTensor = torch.FloatTensor(numpy.array(positions))
moveTensor = torch.LongTensor(numpy.array(moves))


dataset = ChessPositionDataset(boardTensor, moveTensor)
print(f"Done reading game! Read a total of: {len(dataset)} positions.")

pgn.close()
