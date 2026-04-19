import time
import cv2
import numpy as np
import math
import RPi.GPIO as GPIO
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor, servo
from gpiozero import DistanceSensor, TonalBuzzer, LED
from mpu6050 import mpu6050

# =========================
# 1. CONFIGURATION MATERIELLE
# =========================

# --- LEDs et buzzer ---
led1 = LED(11)
led2 = LED(25)
led1.off()
led2.off()
tb = TonalBuzzer(18)

# --- Capteurs de ligne ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN)  # Droite
GPIO.setup(27, GPIO.IN)  # Centre
GPIO.setup(22, GPIO.IN)  # Gauche

# --- I2C & PCA9685 ---
i2c = busio.I2C(SCL, SDA)
pwm = PCA9685(i2c, address=0x5f)
pwm.frequency = 50

# --- Moteurs DC ---
MOTOR_M1_IN1 = 15
MOTOR_M1_IN2 = 14
MOTOR_M2_IN1 = 12
MOTOR_M2_IN2 = 13
motor1 = motor.DCMotor(pwm.channels[MOTOR_M1_IN1], pwm.channels[MOTOR_M1_IN2])
motor2 = motor.DCMotor(pwm.channels[MOTOR_M2_IN1], pwm.channels[MOTOR_M2_IN2])
motor1.decay_mode = motor.SLOW_DECAY
motor2.decay_mode = motor.SLOW_DECAY

# --- Servos ---
servos = {
    i: servo.Servo(pwm.channels[i], min_pulse=500, max_pulse=2400, actuation_range=180)
    for i in range(16)
}

def setangle(ID, alpha):
    servos[ID].angle = alpha

# --- Capteur ultrason ---
Tr = 23
Ec = 24
sensor = DistanceSensor(echo=Ec, trigger=Tr, max_distance=2)

def checkdist():
    return sensor.distance * 100  # cm

# --- Gyroscope MPU6050 ---
gyro = mpu6050(0x68)
angle_z = 0
last_time = time.time()

def get_angle_z():
    global angle_z, last_time
    now = time.time()
    dt = now - last_time
    last_time = now
    gz = gyro.get_gyro_data()['z']  # °/s
    angle_z += gz * dt
    return angle_z

# =========================
# 2. FONCTIONS MOUVEMENT
# =========================
def avancer(speed):
    motor1.throttle = speed
    motor2.throttle = speed

def reculer(speed):
    motor1.throttle = -speed
    motor2.throttle = -speed

def arret():
    motor1.throttle = 0
    motor2.throttle = 0

def tourner_droite(angle):
    setangle(0, 90 + angle)

def tourner_gauche(angle):
    setangle(0, 90 - angle)

# =========================
# 3. CARTOGRAPHIE VISUELLE
# =========================
cap = cv2.VideoCapture(0)
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

map_size = 600
trajectory = np.zeros((map_size, map_size, 3), dtype=np.uint8)
x, y = map_size // 2, map_size // 2
scale = 0.3

# Première image
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_kp, prev_des = orb.detectAndCompute(prev_gray, None)

# =========================
# 4. BOUCLE PRINCIPALE
# =========================
try:
    while True:
        # --- Évitement d'obstacles ---
        dist = checkdist()
        if dist < 30:  # obstacle à moins de 30 cm
            arret()
            led1.on()
            tb.play("C4")
            time.sleep(0.2)
            tb.stop()
            tourner_droite(45)
            time.sleep(0.5)
            continue
        else:
            led1.off()
            avancer(0.1)

        # --- Lecture caméra ---
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)

        if des is not None and prev_des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda m: m.distance)

            if len(matches) > 15:
                dx = dy = 0
                for m in matches[:30]:
                    (x1, y1) = prev_kp[m.queryIdx].pt
                    (x2, y2) = kp[m.trainIdx].pt
                    dx += (x2 - x1)
                    dy += (y2 - y1)
                dx /= 30
                dy /= 30

                x -= int(dx * scale)
                y -= int(dy * scale)
                x = max(0, min(map_size - 1, x))
                y = max(0, min(map_size - 1, y))

                cv2.circle(trajectory, (x, y), 2, (0, 255, 0), -1)

        # --- Affichage ---
        cv2.imshow("Camera", frame)
        cv2.imshow("Carte", trajectory)

        prev_gray = gray
        prev_kp, prev_des = kp, des

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    arret()
    cap.release()
    cv2.destroyAllWindows()
