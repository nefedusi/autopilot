#include <AFMotor.h>
AF_DCMotor motorL(1);
AF_DCMotor motorR(2);
int cmd;
int prev_cmd = 0;
int motor = 0; //0 for neutral, -1 for left, 1 for right
int sign = 0;
int spd = 0;
int num_of_digits = 0;

//ADVANTAGES:
//momental command executing
//defended from hackers :-) (from all inpredictable strings)

//DISADVANTAGES:
//non-beautiful work with motors (if motor == 1) {...}
//code duplication
//difficult code

void setup() {
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    cmd = Serial.read();
    Serial.print("char='");
    Serial.print((char)cmd);
    Serial.print("', int=");
    Serial.println(cmd);
    
    if (cmd == 'L') {
      motor = -1;
    } else if (cmd == 'R') {
      motor = 1;
    } else if (cmd == '+' && (prev_cmd == 'L' || prev_cmd == 'R')) {
      sign = 1;
      num_of_digits = 0;
      spd = 0;
    } else if (cmd == '-' && (prev_cmd == 'L' || prev_cmd == 'R')) {
      sign = -1;
      num_of_digits = 0;
      spd = 0;
    } else if (isdigit(cmd) && (prev_cmd == '+' || prev_cmd == '-' || (isdigit(prev_cmd) && num_of_digits < 3))) { //num_of_digits is not necessary here!
      num_of_digits = num_of_digits + 1;
      spd = spd * 10;
      spd = spd + cmd - 48;
      if (num_of_digits == 3)
      {
        if (motor == -1)
        {
          if (sign == 1)
          {
            motorL.run(FORWARD);
          } else
          {
            motorL.run(BACKWARD);
          }
          motorL.setSpeed(spd);
        } else if (motor == 1)
        {
          if (sign == 1)
          {
            motorR.run(FORWARD);
          } else
          {
            motorR.run(BACKWARD);
          }
          motorR.setSpeed(spd);
        }
      }
    }
    prev_cmd = cmd;
  }
}
