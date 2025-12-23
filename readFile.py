from tensors import boardToTensor, moveToValue
from dataset import ChessPositionDataset
import chess.pgn
import torch
import numpy

pgn = open("dataset/lichess_elite_2020-06.pgn")


positions = []
moves = []
currentGame = 0
maxGames = 10
quarterGames = maxGames // 4
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
    try:
        if (i+1) % quarterGames == 0:
            print(f"Read {((i+1) / maxGames) * 100}% of total games.")
    except ZeroDivisionError:
        pass

print("Converting to tensors...")
boardTensor = torch.FloatTensor(numpy.array(positions))
moveTensor = torch.LongTensor(numpy.array(moves))
print("Done converting to tensors!")

dataset = ChessPositionDataset(boardTensor, moveTensor)
print(f"Done reading games! Read a total of: {len(dataset)} positions.")

pgn.close()
