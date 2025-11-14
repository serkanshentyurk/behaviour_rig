#include "limits.h"
// Pin IDs
const int l_input = 5;
const int r_input = 4;
const int rv_cont = 9;
const int rp_cont = 12;
const int lv_cont = 10;
const int lp_cont = 2;
const int mask_cont = 11;
// Digital input values
volatile bool l_input_val;
volatile bool r_input_val;
// Commands
const char rvalve_on = 'a';
const char rvalve_pulse = 'A';
const char rvalve_off= 'b';
const char lvalve_on = 'c';
const char lvalve_pulse = 'C';
const char lvalve_off= 'd';
const char rpuff_on  = 'e';
const char rpuff_pulse = 'E';
const char rpuff_off = 'f';
const char lpuff_on  = 'g';
const char lpuff_pulse = 'G';
const char lpuff_off = 'h';
const char mask_on = 'm';
const char mask_off = 'n';
const char set_time  = 'S';
char command;
// Pulse variables (default 50 milliseconds
int rvalve_dur = 50;
unsigned long rvalve_start;
bool rvalve_active = false;

int lvalve_dur = 50;
unsigned long lvalve_start;
bool lvalve_active = false;

int rpuff_dur  = 50;
unsigned long rpuff_start;
bool rpuff_active = false;

int lpuff_dur  = 50;
unsigned long lpuff_start;
bool lpuff_active = false;

// Other variables
byte serByte[2];
unsigned long cur_time;
bool report_l_input;
bool l_state = false;
bool report_r_input;
bool r_state = false; 
void setup() {
  pinMode(l_input,INPUT);
  pinMode(r_input,INPUT );
  attachInterrupt(digitalPinToInterrupt(l_input), read_l_input, CHANGE);
  attachInterrupt(digitalPinToInterrupt(r_input), read_r_input, CHANGE);
  pinMode(rv_cont,OUTPUT);
  digitalWrite(rv_cont,false);
  pinMode(rp_cont,OUTPUT);
  digitalWrite(rp_cont,false);
  pinMode(lv_cont,OUTPUT);
  digitalWrite(lv_cont,false);
  pinMode(lp_cont,OUTPUT);
  digitalWrite(lv_cont,false);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()>0){
    Serial.readBytes(serByte,2);
    command = serByte[0];
    switch (command) {
      // Right valve commands
      case rvalve_on:
        digitalWrite(rv_cont,true);
        send_message("Right valve",true);
        break;
      case rvalve_pulse:
        digitalWrite(rv_cont,true);
        send_message("Right valve",true);
        rvalve_active = true;
        rvalve_start = micros();
        break;
      case rvalve_off:
        set_rvalve_off();
        break;
        
      // Left valve commands
      case lvalve_on:
        digitalWrite(lv_cont,true);
        send_message("Left valve",true);
        break;
      case lvalve_pulse:
        digitalWrite(lv_cont,true);
        send_message("Left valve",true);
        lvalve_active = true;
        lvalve_start = micros();
        break;
      case lvalve_off:
        set_lvalve_off();
        break;
        
      // Right puff commands
      case rpuff_on:
        digitalWrite(rp_cont,true);
        send_message("Right puff",true);
        break;
      case rpuff_pulse:
        digitalWrite(rp_cont,true);
        send_message("Right puff",true);
        rpuff_active = true;
        rpuff_start = micros();
        break;
      case rpuff_off:
        set_rpuff_off();
        break;
        
      // Left puff commands
      case lpuff_on:
        digitalWrite(lp_cont,true);
        send_message("Left puff",true);
        break;
      case lpuff_pulse:
        digitalWrite(lp_cont,true);
        send_message("Left puff",true);
        lpuff_active = true;
        lpuff_start  = micros();
        break;
      case lpuff_off:
        set_lpuff_off();
        break;

      // Mask commands
      case mask_on:
        digitalWrite(mask_cont,true);
        send_message("Masking", true);
        break;
      case mask_off:
        digitalWrite(mask_cont,false);
        send_message("Masking", false);
        break;

      // Change pulse durations
      case set_time:
        if (Serial.available() < 5){
          send_message("Invalid time provided.  Must be an integer",false);
        }
      default:
        send_message("Unrecognised control signal",false);
        break;
    }
  }
  cur_time = micros();
  // Check if turning off a pulse
  if (rvalve_active && (pulse_dur(rvalve_start,cur_time) >= rvalve_dur)){
    set_rvalve_off();
  }
  if (lvalve_active && (pulse_dur(lvalve_start,cur_time) >= lvalve_dur)){
    set_lvalve_off();
  }
  if (lpuff_active && (pulse_dur(lpuff_start,cur_time) >= lpuff_dur)){
    set_lpuff_off();
  }
  if (rpuff_active && (pulse_dur(rpuff_start,cur_time) >= rpuff_dur)){
    set_rpuff_off();
  }
  

  // Check if reporting a digital input
  if (report_l_input){
    send_message("Left input",l_state);
    report_l_input = false;
  }
  if (report_r_input){
    send_message("Right input",r_state);
    report_r_input = false;
  }
}

void read_l_input() {
  l_state = digitalRead(l_input);
  report_l_input = true;
}

void read_r_input() {
  r_state = digitalRead(r_input);
  report_r_input = true;
}
void send_message(char message[], bool state){
  Serial.println(message);
  Serial.println(state);
  Serial.println(millis());
}
int pulse_dur(unsigned long pulse_start,unsigned long cur_time){
  if (cur_time > pulse_start){
    return (int)((cur_time - pulse_start)/1000.0);
  } else {
    return (int)((cur_time - (pulse_start - ULONG_MAX))/1000.0);
  }
}
// Turning stuff off
void set_rvalve_off(){
  digitalWrite(rv_cont,false);
  send_message("Right valve",false);
  rvalve_active = false; 
}
void set_lvalve_off(){
  digitalWrite(lv_cont,false);
  send_message("Left valve",false);
  lvalve_active = false; 
}
void set_rpuff_off(){
  digitalWrite(rp_cont,false);
  send_message("Right puff",false);
  rpuff_active = false; 
}
void set_lpuff_off(){
  digitalWrite(lp_cont,false);
  send_message("Left puff",false);
  lpuff_active = false; 
}
