const int ledPin12 = 12;
const int ledPin11 = 11;

unsigned long prevMillis12 = 0;
unsigned long prevMillis11 = 0;

bool flashing12 = false;
bool flashing11 = false;

bool ramping12 = false;
bool ramping11 = false;

unsigned long rampStart12 = 0;
unsigned long rampStart11 = 0;

void setup() {
  pinMode(ledPin12, OUTPUT);
  pinMode(ledPin11, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  char ch;

  if (Serial.available() > 0) {
    ch = Serial.read();
    switch (ch) {
      case 'f':
        flashing12 = true;
        ramping12 = false;
        break;
      case 'm':
        flashing11 = true;
        ramping11 = false;
        break;
      case 'r':
        flashing12 = false;
        ramping12 = true;
        rampStart12 = millis();
        break;
      case 't':
        flashing11 = false;
        ramping11 = true;
        rampStart11 = millis();
        break;
      case 's':
        flashing12 = false;
        ramping12 = false;
        digitalWrite(ledPin12, LOW);
        break;
      case 'k':
        flashing11 = false;
        ramping11 = false;
        digitalWrite(ledPin11, LOW);
        break;
    }
  }

  if (flashing12 && millis() - prevMillis12 >= 12.5) {
    digitalWrite(ledPin12, !digitalRead(ledPin12));
    prevMillis12 = millis();
  }

  if (flashing11 && millis() - prevMillis11 >= 12.5) {
    digitalWrite(ledPin11, !digitalRead(ledPin11));
    prevMillis11 = millis();
  }

  if (ramping12) {
    float t = (millis() - rampStart12) / 300.0;
    if (t >= 1.0) {
      digitalWrite(ledPin12, LOW);
      ramping12 = false;
    } else {
      unsigned long onTime = 12.5 - (12.5 * t);
      unsigned long offTime = 12.5 + (12.5 * t);
      if (digitalRead(ledPin12) == HIGH) {
        if (millis() - prevMillis12 >= onTime) {
          digitalWrite(ledPin12, LOW);
          prevMillis12 = millis();
        }
      } else if (millis() - prevMillis12 >= offTime) {
        digitalWrite(ledPin12, HIGH);
        prevMillis12 = millis();
      }
    }
  }

  if (ramping11) {
    float t = (millis() - rampStart11) / 300.0;
    if (t >= 1.0) {
      digitalWrite(ledPin11, LOW);
      ramping11 = false;
    } else {
      unsigned long onTime = 12.5 - (12.5 * t);
      unsigned long offTime = 12.5 + (12.5 * t);
      if (digitalRead(ledPin11) == HIGH) {
        if (millis() - prevMillis11 >= onTime) {
          digitalWrite(ledPin11, LOW);
          prevMillis11 = millis();
        }
      } else if (millis() - prevMillis11 >= offTime) {
        digitalWrite(ledPin11, HIGH);
        prevMillis11 = millis();
      }
    }
  }
}




