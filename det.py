
import pygame          # Para crear el juego (ventana, imágenes, sonidos, etc.)
import cv2             # OpenCV: para usar la cámara y procesar imágenes
import mediapipe as mp # Mediapipe: detecta la mano y sus puntos (dedos, nudillos...)
import numpy as np     # Para manipular arreglos (aquí, rotar la imagen de la cámara)

# ============================================================
# INICIALIZACIÓN DE MEDIAPIPE
# ============================================================
mp_hands = mp.solutions.hands          # Módulo de detección de manos
mp_draw = mp.solutions.drawing_utils   # Utilidad para dibujar los puntos de la mano
pygame.init()                          # Inicia todos los módulos de pygame

# ============================================================
# CONFIGURACIÓN DE LA VENTANA
# ============================================================
WIDTH= 800
HEIGHT = 600     

screen = pygame.display.set_mode((WIDTH, HEIGHT))   # Crea la ventana
pygame.display.set_caption("Juego Rompe Bloques")   # Título de la ventana
clock = pygame.time.Clock()                         # Reloj para controlar los FPS
running = True                                      # Variable que mantiene el juego activo

# ============================================================
# PALETA (la barra que controlas con el dedo)
# ============================================================
paddle_image = pygame.image.load('paddle.png')                  # Carga la imagen de la paleta
paddle_image = pygame.transform.scale(paddle_image, (150,40))   # La redimensiona a 150x40
paddle_x, paddle_y = WIDTH // 2, HEIGHT - 100   # Posición inicial: centro, cerca de abajo
paddle_width, paddle_height = 150, 40           # Tamaño de la paleta (para las colisiones)
paddle_speed = 10                               # Velocidad (no se usa, porque la mueve tu dedo)

# ============================================================
# PELOTA
# ============================================================
ball_image = pygame.image.load('esfera.png')                    # Carga la imagen de la pelota
ball_image = pygame.transform.scale(ball_image, (40,40))        # La redimensiona a 40x40
ball_x, ball_y = WIDTH // 2, HEIGHT // 2        # Posición inicial: centro de la pantalla
ball_speed_x, ball_speed_y = 8, 8               # Velocidad en X e Y (píxeles por frame)
ball_radius = 10                                # Radio usado para las colisiones

# ============================================================
# BLOQUES
# ============================================================
block_image = pygame.image.load('ladri.png')                    # Carga la imagen del ladrillo
block_rows= 4          # 4 filas
block_columns = 8      # columnas
block_width, block_height = WIDTH // block_columns, 50          # Ancho = 800/8 = 100, alto = 50
block_image = pygame.transform.scale(block_image, (block_width, block_height))

blocks = []   # Lista donde se guardarán todos los bloques (como rectángulos)

# Dos ciclos anidados para crear la cuadrícula de bloques
for row in range(block_rows):
    for col in range(block_columns):
        blocks.append(pygame.Rect(col * block_width,    # Posición X del bloque
                                  row * block_height,   # Posición Y del bloque
                                  block_width,          # Ancho
                                  block_height))        # Alto

# ============================================================
# CÁMARA
# ============================================================
cap = cv2.VideoCapture(0)   # Abre la cámara (0 = cámara por defecto)

# ============================================================
# BUCLE PRINCIPAL DEL JUEGO
# ============================================================
# Configura Mediapipe: confianza mínima de detección/seguimiento y solo 1 mano
with mp_hands.Hands(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    max_num_hands=1) as hands:

    while running:
        # --- Leer un cuadro de la cámara ---
        ret, frame = cap.read()     # ret = True si se leyó bien, frame = la imagen
        if not ret:                 # Si falla la cámara, sale del juego
            break

        # --- Procesar la imagen para detectar la mano ---
        frame = cv2.flip(frame, 1)                       # Voltea la imagen como espejo
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV usa BGR, Mediapipe necesita RGB
        result = hands.process(rgb_frame)                # Detecta la mano en la imagen

        # --- Mover la paleta con el dedo índice ---
        if result.multi_hand_landmarks:                  # Si se detectó alguna mano
            for hand_landmarks in result.multi_hand_landmarks:
                index_finger = hand_landmarks.landmark[8]   # Punto 8 = punta del dedo índice
                # index_finger.x va de 0 a 1, se multiplica por WIDTH para pasarlo a píxeles
                # Se resta la mitad de la paleta para que quede centrada en el dedo
                paddle_x = int(index_finger.x * WIDTH) - paddle_width // 2
                # Dibuja los puntos y líneas de la mano sobre la imagen
                mp_draw.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Mover la pelota ---
        ball_x += ball_speed_x
        ball_y += ball_speed_y

        # --- Rebote en paredes izquierda y derecha ---
        if ball_x - ball_radius < 0 or ball_x + ball_radius > WIDTH:
            ball_speed_x *= -1      # Invierte la dirección horizontal

        # --- Rebote en el techo ---
        if ball_y - ball_radius < 0:
            ball_speed_y *= -1      # Invierte la dirección vertical

        # --- Si la pelota cae al fondo, pierdes ---
        if ball_y + ball_radius > HEIGHT:
            running = False         # Termina el juego

        # --- Limitar la paleta para que no salga de la pantalla ---
        if paddle_x < 0:
            paddle_x = 0
        if paddle_x + paddle_width > WIDTH:
            paddle_x = WIDTH - paddle_width

        # --- Rebote de la pelota con la paleta ---
        # Si la pelota está horizontalmente sobre la paleta y llegó a su altura
        if paddle_x < ball_x < paddle_x + paddle_width and ball_y + ball_radius >= paddle_y:
            ball_speed_y *= -1

        # --- Colisión con los bloques ---
        for block in blocks:
            if block.collidepoint(ball_x, ball_y):   # Si el centro de la pelota toca el bloque
                blocks.remove(block)                 # Elimina el bloque
                ball_speed_y *= -1                   # Rebota
                break                                # Solo rompe un bloque por frame

        # --- Preparar la imagen de la cámara para mostrarla en pygame ---
        rgb_frame = cv2.flip(rgb_frame, 1)                  # Voltea de nuevo (ver error #2 abajo)
        rgb_frame = cv2.resize(rgb_frame, (WIDTH, HEIGHT))  # Ajusta al tamaño de la ventana
        rgb_frame = np.rot90(rgb_frame)                     # Rota (pygame usa ejes distintos a OpenCV)
        rgb_frame = pygame.surfarray.make_surface(rgb_frame)  # Convierte el arreglo en superficie de pygame
        screen.blit(rgb_frame, (0, 0))                      # Dibuja la cámara como fondo

        screen.blit(rgb_frame, (0, 0))                      # (Repetido, no hace falta)
        screen.blit(paddle_image, (paddle_x, paddle_y))     # Dibuja la paleta
        screen.blit(ball_image, (ball_x - ball_radius,      # Dibuja la pelota
                                 ball_y - ball_radius))

        # --- Dibujar todos los bloques que quedan ---
        for block in blocks:
            screen.blit(block_image, (block.x, block.y))

        pygame.display.flip()   # Actualiza la pantalla con todo lo dibujado
        clock.tick(30)          # Limita el juego a 30 FPS

        # --- Salir con la tecla ESC ---
        # (Ojo: cv2.waitKey solo detecta teclas si hay una ventana de OpenCV abierta)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# ============================================================
# LIMPIEZA AL TERMINAR
# ============================================================
cap.release()           # Libera la cámara
cv2.destroyAllWindows() # Cierra ventanas de OpenCV