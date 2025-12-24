import pygame
import chess

darkColor = (181, 135, 99)
lightColor = (240, 218, 181)
selectedColor = (201, 255, 253)

board = chess.Board()
pygame.init()
screen = pygame.display.set_mode((512, 512))
clock = pygame.time.Clock()
running = True
selected = False
selectedCoord = None

whitePawn = pygame.image.load(r"sprites\whitePawn.png")
whiteKnight = pygame.image.load(r"sprites\whiteKnight.png")
whiteBishop = pygame.image.load(r"sprites\whiteBishop.png")
whiteRook = pygame.image.load(r"sprites\whiteRook.png")
whiteQueen = pygame.image.load(r"sprites\whiteQueen.png")
whiteKing = pygame.image.load(r"sprites\whiteKing.png")

blackPawn = pygame.image.load(r"sprites\blackPawn.png")
blackKnight = pygame.image.load(r"sprites\blackKnight.png")
blackBishop = pygame.image.load(r"sprites\blackBishop.png")
blackRook = pygame.image.load(r"sprites\blackRook.png")
blackQueen = pygame.image.load(r"sprites\blackQueen.png")
blackKing = pygame.image.load(r"sprites\blackKing.png")

pieceToImage = {'P' : whitePawn, 'N' : whiteKnight, 'B' : whiteBishop, 'R' : whiteRook, 'Q' : whiteQueen, 'K' : whiteKing, 'p' : blackPawn, 'n' : blackKnight, 'b' : blackBishop, 'r' : blackRook, 'q' : blackQueen, 'k' : blackKing}
findFile = {0 : 'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
findFileReverse = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}


def drawSquares():
    for row in range(8):
        for col in range(8):
            if(row+col)%2 == 0: #Light square
                pygame.draw.rect(screen, lightColor, (row*64, col*64, 64, 64))
            else:
                pygame.draw.rect(screen, darkColor, (row*64, col*64, 64, 64))

def drawPieces(board):
    for row  in range(8):
        for col in range(8):
            square = chess.square(col,7-row)
            piece = board.piece_at(square)
            if piece is None:
                continue
            screen.blit(pieceToImage[piece.symbol()],(col*64, row*64))

def movePiece(coordinates):
    global selected, selectedCoord
    file = findFile[coordinates[0]//64]
    rank = str(8-coordinates[1]//64)

    notation = f"{file}{rank}"


    if (not selected):
        selectedCoord = notation
    elif (selectedCoord == notation):
        return
    else:
        print(f"Move {selectedCoord} to {notation}")
        move = chess.Move.from_uci(f"{selectedCoord}{notation}")
        if(checkLegal(move)):
            board.push(move)
        if board.is_checkmate():
            print("Game over!")


    selected = not selected

def checkLegal(move):
    if (move in board.legal_moves):
        return True
    return False

        
def drawHighlight(selectedCoord, selected):
    if selectedCoord == None or not selected:
        return
    file = findFileReverse[selectedCoord[0]]
    rank = 8-int(selectedCoord[1])
    pygame.draw.rect(screen, selectedColor, (64*file, 64*rank, 64, 64))





while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            movePiece(pygame.mouse.get_pos())

    drawSquares()
    drawHighlight(selectedCoord, selected)
    drawPieces(board)

    pygame.display.flip()

    clock.tick(60) 

pygame.quit()