#include <Servo.h>

int STATE;
int READ = 0;
int MOVE = 1;
int WRITE = 2;

Servo servo_h;
Servo servo_v;
int servo_h_pin = 5;
int servo_v_pin = 6;

int x;
int y;

int x_curr;
int y_curr;
int x_next;
int y_next;

void setup()
{
 STATE = READ;
 x = 0;
 y = 0;
 servo_h.attach(servo_h_pin);
 servo_v.attach(servo_v_pin);
 pinMode(servo_h_pin, OUTPUT);
 pinMode(servo_v_pin, OUTPUT);
 
 Serial.begin(9600);
}

void loop()
{
 if(STATE == READ){
   if(Serial.available() > 0){
     int inByte = Serial.read();
     x = inByte >> 4;
     y = inByte & 15;
     
     x_curr = servo_h.read();
     y_curr = servo_v.read();
     x_next = map(x, 0, 15, 0, 180);
     y_next = map(y, 0, 15, 0, 180);
     
     STATE = MOVE;
     
     //servo_h.write(x_pos);
     //servo_v.write(y_pos);
     //while((servo_h  .read() != x_pos) || (servo_v.read() != y_pos)) ;
     //Serial.write(55);
   }
 }
 
 if(STATE == MOVE){
   if(x_next < x_curr) x_curr--;
   else if(x_next > x_curr) x_curr++;
   if(y_next < y_curr) y_curr--;
   else if(y_next > y_curr) y_curr++;
   servo_h.write(x_curr);
   servo_v.write(y_curr);
   delay(3);
   if((x_curr == x_next) && (y_curr == y_next)) STATE = WRITE;
 }
 
 if(STATE == WRITE){
   Serial.write(55);
   STATE = READ;
 }
}
