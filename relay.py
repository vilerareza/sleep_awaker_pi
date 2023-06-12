import RPi.GPIO as GPIO

class Relay():

    def __init__(self, pin) -> None:
        self.relayPin = pin
        self.relay_init()

    def relay_init(self):
        # GPIO is in BCM mode
        #GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.relayPin, GPIO.OUT)

    def on(self):
        GPIO.output(self.relayPin,1)

    def off(self):
        GPIO.output(self.relayPin,0)

    def cleanup(self):
        GPIO.cleanup()