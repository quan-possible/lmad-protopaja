import Jetson.GPIO as IO

IO.setmode(IO.BOARD)

pinA = 12
pinB = 13

IO.setup(pinA, IO.OUT)
IO.setup(pinB, IO.OUT)

pwmA = IO.PWM(pinA, 100)
pwmB = IO.PWM(pinB, 100)

def control(angle, speed):
	if(angle < 0):
		pwmA.ChangeDutyCycle(speed*(90+angle)/90)
		pwmB.ChangeDutyCycle(speed)
	elif(angle > 0):
		pwmA.ChangeDutyCycle(speed)
		pwmB.ChangeDutyCycle(speed*(90-angle)/90)
	else:
		pwmA.ChangeDutyCycle(speed)
		pwmB.ChangeDutyCycle(speed)
