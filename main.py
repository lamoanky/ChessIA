import pygame
import chess

darkColor = (181, 135, 99)
lightColor = (240, 218, 181)
selectedColor = (201, 255, 253)
mateColor = (255, 0, 0)
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

darkSquare = pygame.image.load(r"sprites\darkSquare.png")
lightSquare = pygame.image.load(r"sprites\lightSquare.png")
selectedSquare = pygame.image.load(r"sprites\selectedSquare.png")
mateSquare = pygame.image.load(r"sprites\mateSquare.png")

board = chess.Board()
pygame.init()
pygame.display.set_icon(whitePawn)
screen = pygame.display.set_mode((512, 512))
pygame.display.set_caption("Chess MonkeyBot")
clock = pygame.time.Clock()
running = True
selected = False
selectedCoord = None
gameOver = None


pieceToImage = {'P' : whitePawn, 'N' : whiteKnight, 'B' : whiteBishop, 'R' : whiteRook, 'Q' : whiteQueen, 'K' : whiteKing, 'p' : blackPawn, 'n' : blackKnight, 'b' : blackBishop, 'r' : blackRook, 'q' : blackQueen, 'k' : blackKing}
findFile = {0 : 'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
findFileReverse = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}


def drawSquares():
    for row in range(8):
        for col in range(8):
            if(row+col)%2 == 0: #Light square
                screen.blit(lightSquare, (row*64, col*64))
                #pygame.draw.rect(screen, lightColor, (row*64, col*64, 64, 64))
            else:
                screen.blit(darkSquare, (row*64, col*64))
                #pygame.draw.rect(screen, darkColor, (row*64, col*64, 64, 64))

def drawPieces(board):
    for row  in range(8):
        for col in range(8):
            square = chess.square(col,7-row)
            piece = board.piece_at(square)
            if piece is None:
                continue
            screen.blit(pieceToImage[piece.symbol()],(col*64, row*64))

def movePiece(coordinates):
    global selected, selectedCoord, gameOver
    file = findFile[coordinates[0]//64]
    rank = str(8-coordinates[1]//64)

    notation = f"{file}{rank}"
    piece = board.piece_at(chess.parse_square(notation))


    if (not selected):
        selectedCoord = notation
    elif (selectedCoord == notation):
        return
    else:
        move = chess.Move.from_uci(f"{selectedCoord}{notation}")
        promoteMove = chess.Move.from_uci(f"{selectedCoord}{notation}q")
        if(checkLegal(move, piece, promoteMove) == 1):
            board.push(move)
        elif (checkLegal(move, piece, promoteMove) == 2):
            board.push(promoteMove)


        if board.is_game_over():
            if board.result() == "1-0":
                gameOver = "Black"
                print("White wins!")
            elif board.result() == "0-1":
                gameOver = "White"
                print("Black wins!")
            else:
                gameOver = "Draw"
                print("Draw!")


    selected = not selected

def checkLegal(move, piece, promoteMove):
    
    if (move in board.legal_moves):
        return 1
    elif promoteMove in board.legal_moves:
        return 2
    return 0

        
def drawHighlight(selectedCoord, selected):
    if selectedCoord == None or not selected:
        return
    file = findFileReverse[selectedCoord[0]]
    rank = 8-int(selectedCoord[1])
    screen.blit(selectedSquare, (64*file, 64*rank))
    #pygame.draw.rect(screen, selectedColor, (64*file, 64*rank, 64, 64))

def drawCheckmate():

    if gameOver == None:
        return
    
    if gameOver == "Black":
        square = chess.square_name(board.king(chess.BLACK))
        file = findFileReverse[square[0]]
        rank = 8-int(square[1])
        screen.blit(mateSquare, (64*file, 64*rank))
        #pygame.draw.rect(screen, mateColor, (64*file, 64*rank, 64, 64))
    if gameOver == "White":
        square = chess.square_name(board.king(chess.WHITE))
        file = findFileReverse[square[0]]
        rank = 8-int(square[1])
        screen.blit(mateSquare, (64*file, 64*rank))
        #pygame.draw.rect(screen, mateColor, (64*file, 64*rank, 64, 64))



while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            movePiece(pygame.mouse.get_pos())

    drawSquares()
    drawHighlight(selectedCoord, selected)
    drawCheckmate()

    drawPieces(board)

    pygame.display.flip()

    clock.tick(60) 

pygame.quit()