#define VARIABLE_LABEL "sensor" // Assing the variable label
#define DEVICE_LABEL "esp32" // Assig the device label

#define SENSOR A0 // Set the A0 as SENSOR

// Space to store values to send
char str_sensor[10];

/****************************************
 * Auxiliar Functions
 ****************************************/
/****************************************
 * Main Functions
 ****************************************/
void setup() {
  Serial.begin(115200);
  pinMode(SENSOR, INPUT);
}

void loop() {
  /* 4 is mininum  width, 2 is precision; float value is copied onto str_sensor*/
   float sensor = analogRead(SENSOR); 
 dtostrf(sensor, 4, 2, str_sensor);
Serial.println(analogRead(34));

  delay(20);
}
