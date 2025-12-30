from tensors import boardToTensor, moveToValue
from dataset import ChessPositionDataset
import chess.pgn
import torch
import numpy

class Reader:
    def __init__(self, file):
        self.pgn = open(file)
        self.file = file
        self.gamesParsed = 0
    def readChunk(self, chunkSize):
        positions = []
        moves = []
        start = self.gamesParsed
        print("Initializing chunk...")
        for i in range(chunkSize):
            game = chess.pgn.read_game(self.pgn)
            if game is None:
                break
            board = game.board()

            

            for move in game.mainline_moves():
                boardArray = boardToTensor(board)
                moveValue = moveToValue(move.uci())

                positions.append(boardArray)
                moves.append(moveValue)
                board.push(move)
            self.gamesParsed += 1

        boardTensor = torch.FloatTensor(numpy.array(positions))
        moveTensor = torch.LongTensor(numpy.array(moves))


        dataset = ChessPositionDataset(boardTensor, moveTensor)
        print(f"Done reading chunk! Read from games {start} - {self.gamesParsed}")

        return dataset
    
    def close(self):
        self.pgn.close()

    def restart(self):
        self.pgn.close()
        self.pgn = open(self.file)
        self.gamesParsed = 0



"""
def readF(chunkSize, chunkAmount):
    pgn = open("dataset/lichess_elite_2020-06.pgn")
    positions = []
    moves = []
    print("Initializing game...")

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
    readF(0, 100)
"""

