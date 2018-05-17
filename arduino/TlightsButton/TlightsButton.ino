int r = 2, y = 8, g = 12, btn = 6;
bool pressed = 0;
bool is_green = 1;

void setup() {
  pinMode(r, OUTPUT);
  pinMode(y, OUTPUT);
  pinMode(g, OUTPUT);
  pinMode(btn, INPUT);
}


void loop() {
  pressed = digitalRead(btn);
  if (pressed) {
    delay(50);
    pressed = digitalRead(btn);
    if (pressed) {
      if (is_green) {
        digitalWrite(g, LOW);
        delay(500);
        digitalWrite(g, HIGH);
        delay(500);
        digitalWrite(g, LOW);
        delay(500);
        digitalWrite(g, HIGH);
        delay(500);
        digitalWrite(g, LOW);
        delay(500);
        digitalWrite(g, HIGH);
        delay(500);
        digitalWrite(g, LOW);
        digitalWrite(y, HIGH);
        delay(2000);
        digitalWrite(y, LOW);
        digitalWrite(r, HIGH);
      } else {
        digitalWrite(y, HIGH);
        delay(2000);
        digitalWrite(r, LOW);
        digitalWrite(y, LOW);
        digitalWrite(g, HIGH);
      }
    }
  }  
}
