int r = 2, y = 8, g = 12;

void setup() {
  pinMode(r, OUTPUT);
  pinMode(y, OUTPUT);
  pinMode(g, OUTPUT);
}


void loop() {
  digitalWrite(r, HIGH);
  delay(6000);
  digitalWrite(y, HIGH);
  delay(2000);
  digitalWrite(r, LOW);
  digitalWrite(y, LOW);
  
  digitalWrite(g, HIGH);
  delay(6000);  
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
}
