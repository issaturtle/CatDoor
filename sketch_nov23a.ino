// Example code to control a DC motor with L298N and Arduino Mega 2560

const int analogInputPin = A0;
const int motorEnablePin = A3;  // Analog pin 0
const int motorInput1 = A2;     // motorInput 1 needs to be attached to pin 1 on L289N
const int motorInput2 = A1;     // motorInput 2 needs to be attached to pin 2 on L289N
int hldr =0;

const int threshold = 600;
 int opened; 

void setup() {
  Serial.begin(9600);
  pinMode(motorEnablePin, OUTPUT);
  pinMode(motorInput1, OUTPUT);
  pinMode(motorInput2, OUTPUT);
  pinMode(analogInputPin, INPUT);
  digitalWrite(motorEnablePin, HIGH);  // Enable the motor
  // openAndShut();
  clockwise();
  delay(1000);
  opened=0;
}
//1110
void loop() 
{
  
  if (hldr == 0) // intial value gotten by sensor usually high enough to trip threshold so just grab first and discard
  {
    int sensorValue = analogRead(analogInputPin);
    hldr =1;
    delay(500);
  }
  delay(500);
  int sensorValue = analogRead(analogInputPin);
  //clockwise();
  Serial.println(sensorValue);
  if (sensorValue > threshold)
  {
    //exit;
    //opened=1;
    openAndShut();
    delay(5000);
    }
  if(opened==1)
  {
    Serial.println("opened");
  }
  //13.5 per min
  //4.44 per 15 seconds 
  //.225 turns per second 
  // .225 * X = .25 => X=1.111
  
  //openAndShut();
}
void clockwise()
{
  digitalWrite(motorInput1, LOW);
  digitalWrite(motorInput2, HIGH);
}
void Cclockwise()
{
  digitalWrite(motorInput1, HIGH);
  digitalWrite(motorInput2, LOW);
}
void neut()
{
  digitalWrite(motorInput1, LOW);
  digitalWrite(motorInput2, LOW);
}
void turnQ()
{  // Move the motor forward
 
   digitalWrite(motorInput1, LOW);
  digitalWrite(motorInput2, LOW);
  delay(1000);

  digitalWrite(motorInput1, HIGH);
  digitalWrite(motorInput2, LOW);
  delay(2210);

  }
void openAndShut()
{  
  
  //bias when turning counter clockwise. testing 10 less ms when turning in direction
  neut();
  delay(1000);
  //delay(5000); // testing wait time 
  
  Cclockwise();
  delay(2000);
  //delay(3000);// having a time opening door could be motor could be door. testing longer time to oopen
 
  neut();
  delay(10000);
 
  clockwise();
  delay(2000);
  neut();
}