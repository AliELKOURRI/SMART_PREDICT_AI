# ============================================================
# robot/robot_controller.py
# Contrôleur physique du robot sur Raspberry Pi
# Traduit les chemins A* en commandes GPIO moteurs
# Phase 4 : Action (cf. rapport §2.7.3)
# ============================================================

import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app_config.settings import (
    GPIO_MOTOR_LEFT_FORWARD, GPIO_MOTOR_LEFT_BACKWARD,
    GPIO_MOTOR_RIGHT_FORWARD, GPIO_MOTOR_RIGHT_BACKWARD,
    GPIO_PWM_LEFT, GPIO_PWM_RIGHT,
    ROBOT_VITESSE_SEC
)

# Import conditionnel GPIO (disponible uniquement sur Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    SUR_RASPBERRY = True
    print("[ROBOT] GPIO Raspberry Pi détecté.")
except ImportError:
    SUR_RASPBERRY = False
    print("[ROBOT] Mode simulation (GPIO non disponible).")


# ──────────────────────────────────────────────────────────────
# Gestion des directions
# ──────────────────────────────────────────────────────────────
# Directions basées sur le déplacement (dl, dc) dans la grille
DIRECTIONS = {
    (-1,  0): "AVANT",    # ligne diminue → avancer
    ( 1,  0): "ARRIERE",  # ligne augmente → reculer
    ( 0, -1): "GAUCHE",   # colonne diminue → gauche
    ( 0,  1): "DROITE",   # colonne augmente → droite
}


class RobotController:
    """
    Contrôleur du robot physique sur Raspberry Pi.
    En mode simulation, affiche les commandes en console.
    """

    def __init__(self, vitesse_pct: int = 70):
        """
        vitesse_pct : puissance moteur en % (PWM duty cycle)
        """
        self.vitesse     = vitesse_pct
        self.position    = None  # Position actuelle dans la grille
        self.pwm_gauche  = None
        self.pwm_droite  = None
        self._initialiser_gpio()

    def _initialiser_gpio(self):
        """Configure les pins GPIO en mode sortie."""
        if not SUR_RASPBERRY:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        pins = [
            GPIO_MOTOR_LEFT_FORWARD, GPIO_MOTOR_LEFT_BACKWARD,
            GPIO_MOTOR_RIGHT_FORWARD, GPIO_MOTOR_RIGHT_BACKWARD,
            GPIO_PWM_LEFT, GPIO_PWM_RIGHT
        ]
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)

        # PWM à 1000 Hz
        self.pwm_gauche = GPIO.PWM(GPIO_PWM_LEFT, 1000)
        self.pwm_droite = GPIO.PWM(GPIO_PWM_RIGHT, 1000)
        self.pwm_gauche.start(0)
        self.pwm_droite.start(0)
        print("[GPIO] Initialisé.")

    # ─── Commandes moteurs de base ────────────────────────────

    def avancer(self, duree: float = None):
        print("[MOTEUR] ▲ AVANT")
        if SUR_RASPBERRY:
            GPIO.output(GPIO_MOTOR_LEFT_FORWARD,  GPIO.HIGH)
            GPIO.output(GPIO_MOTOR_LEFT_BACKWARD, GPIO.LOW)
            GPIO.output(GPIO_MOTOR_RIGHT_FORWARD, GPIO.HIGH)
            GPIO.output(GPIO_MOTOR_RIGHT_BACKWARD,GPIO.LOW)
            self.pwm_gauche.ChangeDutyCycle(self.vitesse)
            self.pwm_droite.ChangeDutyCycle(self.vitesse)
        if duree:
            time.sleep(duree)
            self.stop()

    def reculer(self, duree: float = None):
        print("[MOTEUR] ▼ ARRIÈRE")
        if SUR_RASPBERRY:
            GPIO.output(GPIO_MOTOR_LEFT_FORWARD,  GPIO.LOW)
            GPIO.output(GPIO_MOTOR_LEFT_BACKWARD, GPIO.HIGH)
            GPIO.output(GPIO_MOTOR_RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(GPIO_MOTOR_RIGHT_BACKWARD,GPIO.HIGH)
            self.pwm_gauche.ChangeDutyCycle(self.vitesse)
            self.pwm_droite.ChangeDutyCycle(self.vitesse)
        if duree:
            time.sleep(duree)
            self.stop()

    def tourner_gauche(self, duree: float = None):
        print("[MOTEUR] ◄ GAUCHE")
        if SUR_RASPBERRY:
            GPIO.output(GPIO_MOTOR_LEFT_FORWARD,  GPIO.LOW)
            GPIO.output(GPIO_MOTOR_LEFT_BACKWARD, GPIO.HIGH)
            GPIO.output(GPIO_MOTOR_RIGHT_FORWARD, GPIO.HIGH)
            GPIO.output(GPIO_MOTOR_RIGHT_BACKWARD,GPIO.LOW)
            self.pwm_gauche.ChangeDutyCycle(self.vitesse)
            self.pwm_droite.ChangeDutyCycle(self.vitesse)
        if duree:
            time.sleep(duree)
            self.stop()

    def tourner_droite(self, duree: float = None):
        print("[MOTEUR] ► DROITE")
        if SUR_RASPBERRY:
            GPIO.output(GPIO_MOTOR_LEFT_FORWARD,  GPIO.HIGH)
            GPIO.output(GPIO_MOTOR_LEFT_BACKWARD, GPIO.LOW)
            GPIO.output(GPIO_MOTOR_RIGHT_FORWARD, GPIO.LOW)
            GPIO.output(GPIO_MOTOR_RIGHT_BACKWARD,GPIO.HIGH)
            self.pwm_gauche.ChangeDutyCycle(self.vitesse)
            self.pwm_droite.ChangeDutyCycle(self.vitesse)
        if duree:
            time.sleep(duree)
            self.stop()

    def stop(self):
        print("[MOTEUR] ■ STOP")
        if SUR_RASPBERRY:
            for pin in [GPIO_MOTOR_LEFT_FORWARD, GPIO_MOTOR_LEFT_BACKWARD,
                        GPIO_MOTOR_RIGHT_FORWARD, GPIO_MOTOR_RIGHT_BACKWARD]:
                GPIO.output(pin, GPIO.LOW)
            self.pwm_gauche.ChangeDutyCycle(0)
            self.pwm_droite.ChangeDutyCycle(0)

    # ─── Exécution d'un chemin A* ─────────────────────────────

    def executer_chemin(self, chemin: list, callback_etape=None):
        """
        Exécute physiquement le chemin calculé par A*.
        chemin : liste de positions (ligne, colonne)
        callback_etape : fonction appelée à chaque étape (pour le dashboard)
        """
        if len(chemin) < 2:
            print("[ROBOT] Déjà à destination.")
            return

        self.position = chemin[0]
        print(f"\n[ROBOT] Mission démarrée — {len(chemin)-1} étapes")
        print(f"  Départ   : {chemin[0]}")
        print(f"  Arrivée  : {chemin[-1]}")

        for i in range(1, len(chemin)):
            pos_prec  = chemin[i - 1]
            pos_actuelle = chemin[i]
            dl = pos_actuelle[0] - pos_prec[0]
            dc = pos_actuelle[1] - pos_prec[1]
            direction = DIRECTIONS.get((dl, dc), "INCONNU")

            print(f"  Étape {i:3d}/{len(chemin)-1} : {pos_prec} → {pos_actuelle}  [{direction}]")

            # Commande moteur
            if direction == "AVANT":
                self.avancer(duree=ROBOT_VITESSE_SEC)
            elif direction == "ARRIERE":
                self.reculer(duree=ROBOT_VITESSE_SEC)
            elif direction == "GAUCHE":
                self.tourner_gauche(duree=ROBOT_VITESSE_SEC)
            elif direction == "DROITE":
                self.tourner_droite(duree=ROBOT_VITESSE_SEC)

            self.position = pos_actuelle

            # Callback pour le dashboard (mise à jour en temps réel)
            if callback_etape:
                callback_etape(etape=i, position=pos_actuelle, total=len(chemin)-1)

            time.sleep(0.05)  # Petite pause entre commandes

        self.stop()
        print(f"[ROBOT] Mission terminée. Position finale : {self.position}\n")

    # ─── Nettoyage GPIO ───────────────────────────────────────

    def nettoyer(self):
        """Libère les ressources GPIO (à appeler à l'arrêt du programme)."""
        self.stop()
        if SUR_RASPBERRY:
            if self.pwm_gauche:
                self.pwm_gauche.stop()
            if self.pwm_droite:
                self.pwm_droite.stop()
            GPIO.cleanup()
            print("[GPIO] Nettoyé.")

    def __del__(self):
        try:
            self.nettoyer()
        except Exception:
            pass