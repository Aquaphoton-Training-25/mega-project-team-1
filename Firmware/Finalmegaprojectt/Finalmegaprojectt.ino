#include <Servo.h>

#define trig_pin 7
#define echo_pin 8

#define right_motor_IN1 2  //IN3
#define right_motor_IN2 4  //IN4
#define right_motor_en 3

#define left_motor_IN1 13  //IN1
#define left_motor_IN2 12  //IN2
#define left_motor_en 11

#define servo_pin 10

#define red_rgbled 5
#define blue_rgbled 6
#define green_rgbled 9

#define current_sensor A0
#define voltage_sensor A1

#define tolerance 2


Servo servo_motor;

float current;  // current sensor reading
float voltage;  //current sensor reading

float duration;
float distance; //distance read from ultrasonic sensor

int speed = 90;  //The default speed for the car is medium speed 

// PID control variables
float setpoint = 20.0;  // desired distance from the wall (in cm)
float kp=1.0;
float ki=0.0;
float kd=0.0;
float previous_error = 0;
float previous_time = 0;
float output = 0;
float lower_limit = 10, upper_limit = 100;  // PID output limits
float dist;                                 //Variable for PID
float pid_output;
float auto_time;  //variable to store time in seconds of enetring autonomous mode
double last_command_time;

String state = "S"; //The car starts at stop state


void setup() {

  analogReference(EXTERNAL);

  Serial.begin(9600);

  servo_motor.attach(servo_pin);

  pinMode(trig_pin, OUTPUT);
  pinMode(echo_pin, INPUT);

  pinMode(right_motor_IN1, OUTPUT);
  pinMode(right_motor_IN2, OUTPUT);
  pinMode(right_motor_en, OUTPUT);

  pinMode(left_motor_IN1, OUTPUT);
  pinMode(left_motor_IN2, OUTPUT);
  pinMode(left_motor_en, OUTPUT);

  pinMode(green_rgbled, OUTPUT);
  pinMode(blue_rgbled, OUTPUT);
  pinMode(red_rgbled, OUTPUT);

  pinMode(current_sensor, INPUT);
  pinMode(voltage_sensor, INPUT);
}

void loop() {

  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\r');  //Reading any data sent from user.
    process_command(command);
    last_command_time = millis();
  }
  // Continue sending readings at regular intervals
 if (millis() - last_command_time > 100) { // Send readings every 100ms
        send_readings();
        last_command_time = millis();
     }
}

//this function is used when we click the autonomous mode in the gui
void autonomous() {
  
  dist = find_wall();// we call the functon find wall to see the nearest distance
  auto_time = millis() / 1000.0;
 /*while loop to see the if we get command s it will break the autonomus mode
 but if nt it will read distance and compute it then control the motors to make the desired distace ==feedback distance*/
  while (1) {
    
    if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\r');  //Reading any data sent from user.
    last_command_time = millis();

      if (command == "S") {
        drive_break();
        break;
      }
    }
    if (millis() - last_command_time > 100) { // Send readings every 100ms
        send_readings();
        last_command_time = millis();
    }
   pid_output= compute_output(dist);
    control_motors(pid_output, speed);
    dist = read_distance();
    }
}

void drive_forward(float enable1, float enable2) {  //Car moves forward
  digitalWrite(right_motor_IN1, HIGH);
  digitalWrite(right_motor_IN2, LOW);
  analogWrite(right_motor_en, enable1);
                                        //Both motors move with the same speed in the forward direction
  digitalWrite(left_motor_IN1, HIGH);
  digitalWrite(left_motor_IN2, LOW);
  analogWrite(left_motor_en, enable2);
}

void drive_backward(float enable1, float enable2) { //Car moves backward
  digitalWrite(right_motor_IN1, LOW);
  digitalWrite(right_motor_IN2, HIGH);
  analogWrite(right_motor_en, enable1);
                                          //Both motors move with the same speed in the backward direction
  digitalWrite(left_motor_IN1, LOW);
  digitalWrite(left_motor_IN2, HIGH);
  analogWrite(left_motor_en, enable2);
}

void turn_right(float enable1, float enable2) { //Car rotates to the right

  digitalWrite(right_motor_IN1, HIGH);
  digitalWrite(right_motor_IN2, LOW);
  analogWrite(right_motor_en, enable1);//The rightt motor moves backward.
                                         //The left motor moves in the opposite direction.
  digitalWrite(left_motor_IN1, LOW);
  digitalWrite(left_motor_IN2, HIGH);
  analogWrite(left_motor_en, enable2);
}

void turn_Left(float enable1, float enable2) {  //Car rotates to the left
  digitalWrite(right_motor_IN1,LOW);
  digitalWrite(right_motor_IN2,HIGH);
  analogWrite(right_motor_en, enable1);  //The rightt motor moves forward.
                                         //The left motor moves in the opposite direction. 
  digitalWrite(left_motor_IN1,HIGH);
  digitalWrite(left_motor_IN2, LOW);
  analogWrite(left_motor_en, enable2);
}


void drive_break() {                    //Car stops    
  digitalWrite(right_motor_IN1, LOW);
  digitalWrite(right_motor_IN2, LOW);
  analogWrite(right_motor_en, 0);
                                      //Sets motors' speeds to 0
  digitalWrite(left_motor_IN1, LOW);
  digitalWrite(left_motor_IN2, LOW);
  analogWrite(left_motor_en, 0);
}

float read_distance() { //get ultrasonic sensor reading
  digitalWrite(trig_pin, LOW);   // ensure trigger pin is LOW
  delayMicroseconds(2);          // stabilize sensor
  digitalWrite(trig_pin, HIGH);  // send 10 microsecond pulse to trigger
  delayMicroseconds(10);
  digitalWrite(trig_pin, LOW);

  duration = pulseIn(echo_pin, HIGH);  // get duration of echo pulse
  distance = duration * 0.03442 / 2;   // convert time to distance in cm


  Serial.print(distance);
  return distance;
}

void process_command(String command) { 

  if (command == "F") {
    drive_forward(speed, speed);
    state = command;  //update the car's state
  } else if (command == "B") {
    drive_backward(speed, speed);  
    state = command;
  } else if (command == "L") {
    turn_Left(speed, speed * 0.75);  
    state = command;
  } else if (command == "R") {
    turn_right(speed * 0.75, speed);  
    state = command;
  } else if (command == "S") {
    drive_break();  // Stop
    state = command;
  } else if (command == "A") {
    autonomous();
    state = command;
  } else if (command == "H")
    speed = 140;
    else if (command == "M")
    speed = 90;
    else if (command == "Q")
    speed = 40;
}

// this function is used to rotate servo motor  that is aatached with ultrasonic sensor to see the nearest wall
float find_wall() {
  float wall_1 = read_distance();
  servo_motor.write(180);
  delay(500);
  float wall_2 = read_distance();
  if (wall_2 > wall_1) { // detects the nearest wall 
    servo_motor.write(0); //if wall1 is nearest it moves the servo back to wall1
    return wall_1;
  }
  return wall_2;
}

/*
we can use constrain function instead of map to limits they both act like each other,
they return 10 if the output is less than 10cm
and 100 if the output is more than hundred cm
and it return the output if it is in range between 10 & 10e
then return the final output
*/
float map_to_limits(float value,float lower_limit,float upper_limit) {
  if (value < lower_limit)
    return lower_limit;
  else if (value > upper_limit)
    return upper_limit;
  else
    return value;
}
//function to calculate the PID output bassd on the feedback from the sensor 
float compute_output(float feedback) {

  float current_time = millis() * 0.001;  // Convert time to seconds
  float error = setpoint - feedback;      // Error = desired distance - actual distance
  float delta_time = current_time - auto_time;//calculate the cahnge in time
  float integral_part = 0;
  float derivative_part = 0;
  float set_output = 0;

  if (delta_time > 0)  // to avoid devision by zero
  {
    integral_part = previous_error + (error * delta_time); //calculate the integral part in the integral term in PID controller
    derivative_part = (error - previous_error) / delta_time;//calculate the derivative part in the derivative term in PID controller

    output = kp * error + ki * integral_part + kd * derivative_part; //PID equation
    set_output = map_to_limits(output,10.0,100.0); /*call the map_to limits and send the output of pid
                                                    and the limits of the distance to set the car away */
    previous_error = error;
    auto_time = current_time;
  }
  return set_output;
}

//this function is used to get the output of the PId then use it to conttrol the spped of the two motors to turn right or left
void control_motors(float pid_output,int speed) {
  
  // adjust the speed of the motors based on the PID output
  float left_motor_speed = speed - pid_output; //the direction of wall may differ
  float right_motor_speed = speed + pid_output;

  // ensure the speeds are within valid PWM range (0-255)
  left_motor_speed = map_to_limits(left_motor_speed, 0, 255.0);
  right_motor_speed = map_to_limits(right_motor_speed, 0, 255.0);
  if (abs(setpoint - dist) <= tolerance  ) { // see the differnce between the desirewd point and the feedback from the sensor
    left_motor_speed = speed;
    right_motor_speed = speed;
  }

  drive_forward(left_motor_speed, right_motor_speed); // Drive the motors forward with adjusted speeds
}
// function to read the current from the current  sensor(ACS712 5A)
void read_current() {
  current = analogRead(current_sensor)/1023.0*5.0; //voltage on sensor pin
  current =(current-2.5)/0.185; //current value
  if (current<0)
    current=0;
  Serial.print(current);
}
//function to read the voltage from a voltage divider and return the real readings for the volt
void read_voltage() {
  voltage=analogRead(voltage_sensor)/1023.0*5.0; //vltage on 1kohm resistor 
  voltage=voltage*(4700.0+1000.0)/1000.0; //Total voltage
  Serial.print(voltage);
}

// this function is used to call the functions of sensors to read them 
void send_readings() {
  read_voltage();
  Serial.print(":");  //send colon's to split data

  read_current();
  Serial.print(":");

  read_distance();
  Serial.print(":");
  
  Serial.println(state); //sends newline character to split data packages.  
}
