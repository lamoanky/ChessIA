from dataset import ChessPositionDataset
import chess.pgn
import torch
import numpy

class Reader:
    def __init__(self, file, tensors):
        self.pgn = open(file)
        self.file = file
        self.gamesParsed = 0
        self.tensors = tensors

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
                boardArray = self.tensors.boardToTensor(board)
                moveValue = self.tensors.moveToValue(move.uci())

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


        