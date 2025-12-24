from tensors import boardToTensor, moveToValue
from dataset import ChessPositionDataset
import chess.pgn
import torch
import numpy



def readF(currentChunk, chunkSize):
    pgn = open("dataset/lichess_elite_2020-06.pgn")
    positions = []
    moves = []
    print("Initializing game...")
    for i in range(chunkSize*currentChunk):
        chess.pgn.read_game(pgn)

    for i in range(chunkSize):
        game = chess.pgn.read_game(pgn)
        board = game.board()

        if game is None:
            break

        for move in game.mainline_moves():
            boardArray = boardToTensor(board)
            moveValue = moveToValue(move.uci())

            positions.append(boardArray)
            moves.append(moveValue)
            board.push(move)
        


    boardTensor = torch.FloatTensor(numpy.array(positions))
    moveTensor = torch.LongTensor(numpy.array(moves))


    dataset = ChessPositionDataset(boardTensor, moveTensor)
    print(f"Done reading games! Read a total of: {len(dataset)} positions.")

    pgn.close()

    return dataset


if __name__ == "__main__":
    readF()
