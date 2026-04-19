import time
import math
import busio
from board import SCL, SDA
from adafruit_pca9685 import PCA9685
from adafruit_motor import motor
from mpu6050 import mpu6050

# --- Initialisation I2C ---
i2c = busio.I2C(SCL, SDA)
pwm = PCA9685(i2c, address=0x5f)  # Adresse I2C du PCA9685
pwm.frequency = 50

# --- Configuration moteurs ---
motor1 = motor.DCMotor(pwm.channels[15], pwm.channels[14])  # Moteur gauche
motor2 = motor.DCMotor(pwm.channels[12], pwm.channels[13])  # Moteur droit
motor1.decay_mode = motor.SLOW_DECAY
motor2.decay_mode = motor.SLOW_DECAY

# --- Gyroscope ---
gyro = mpu6050(0x68)
angle_z = 0.0
last_time = time.time()

# --- PID ---
Kp, Ki, Kd = 1.5, 0.0, 0.4
integral = 0
last_error = 0

# --- Paramètres trajectoire ---
delta_x = 0.1  # m
current_x = 0.0
BASE_SPEED = 0.2  # vitesse de base (20%)

# Fonction mathématique à suivre
def f(x):
    return 0.1 * math.sin(x)  # Exemple : sinusoïde

# --- Filtre complémentaire pour angle Z ---
def get_angle_z():
    global angle_z, last_time
    now = time.time()
    dt = now - last_time
    last_time = now

    accel_data = gyro.get_accel_data()
    gyro_data = gyro.get_gyro_data()

    accel_angle = math.degrees(math.atan2(accel_data['y'], accel_data['x']))
    gz = gyro_data['z']  # °/s

    # Filtre complémentaire
    angle_z = 0.98 * (angle_z + gz * dt) + 0.02 * accel_angle
    return angle_z

# --- Contrôle moteurs ---
def set_motor_speeds(left, right):
    # Clamp entre -1 et 1
    left = max(min(left, 1), -1)
    right = max(min(right, 1), -1)
    motor1.throttle = left
    motor2.throttle = right

# --- Boucle principale ---
try:
    while True:
        # 1. Lire orientation actuelle
        angle_mesure = get_angle_z()

        # 2. Calculer angle cible
        y1 = f(current_x)
        y2 = f(current_x + delta_x)
        theta_cible = math.degrees(math.atan2(y2 - y1, delta_x))

        # 3. PID
        erreur = theta_cible - angle_mesure
        integral += erreur
        derivative = erreur - last_error
        correction = Kp * erreur + Ki * integral + Kd * derivative
        last_error = erreur

        # 4. Ajuster moteurs avec vitesse de base réduite
        left_speed = BASE_SPEED + (correction / 100)
        right_speed = BASE_SPEED - (correction / 100)

        set_motor_speeds(left_speed, right_speed)

        # 5. Avancer dans la trajectoire
        current_x += delta_x
        time.sleep(0.1)

except KeyboardInterrupt:
    pass

finally:
    set_motor_speeds(0, 0)
