// Load Wi-Fi library
#include <WiFi.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <WiFiMulti.h>

// Replace with your network credentials
const char* ssid = "abd";
const char* password = "sherlockedissecure";

// Set web server port number to 80
WiFiServer server(80);

// Variable to store the HTTP request
String header;

unsigned long currentTime = millis();
// Previous time
unsigned long previousTime = 0; 
// Define timeout time in milliseconds (example: 2000ms = 2s)
const long timeoutTime = 2000;
int output = 0;
String myString="";
void setup() {
  Serial.begin(115200);
  // Initialize the output variables as outputs
        output = int(analogRead(A0));
      myString = myString+String(output);
  for(int i =0;i <2365;i++){
        output = int(analogRead(A0));
      myString = myString+","+String(output);

    }
  // Connect to Wi-Fi network with SSID and password
  Serial.print("Connecting to ");
  Serial.println(ssid);
//WiFi.disconnect();
//delay(1);
  WiFi.begin(ssid, password);
    Serial.print(WiFi.status());
 while(WiFi.waitForConnectResult() != WL_CONNECTED){
    delay(500);
    Serial.print(WiFi.status());
    Serial.print(".");
  }
  // Print local IP address and start web server
  Serial.println("");
  Serial.println("WiFi connected.");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
  server.begin();
}

void loop(){
  WiFiClient client = server.available();   // Listen for incoming clients

  if (client) {                             // If a new client connects,
    currentTime = millis();
    previousTime = currentTime;
    Serial.println("New Client.");          // print a message out in the serial port
    String currentLine = "";                // make a String to hold incoming data from the client
    while (client.connected() && currentTime - previousTime <= timeoutTime) {  // loop while the client's connected
      currentTime = millis();
      if (client.available()) {             // if there's bytes to read from the client,
        char c = client.read();             // read a byte, then
        Serial.write(c);                    // print it out the serial monitor
        header += c;
        if (c == '\n') {                    // if the byte is a newline character
          // if the current line is blank, you got two newline characters in a row.
          // that's the end of the client HTTP request, so send a response:
          if (currentLine.length() == 0) {
            // HTTP headers always start with a response code (e.g. HTTP/1.1 200 OK)
            // and a content-type so the client knows what's coming, then a blank line:
            client.println("HTTP/1.1 200 OK");
            client.println("Content-type:text/html");
            client.println("Connection: close");
            client.println();
            
            client.println("<!DOCTYPE html><html>");
            
            client.println("<body>");
            
            client.println(myString+"</body>");
            client.println();
            client.stop();
            break;
          }else { // if you got a newline, then clear currentLine
            currentLine = "";
          }
        } else if (c != '\r') {  // if you got anything else but a carriage return character,
          currentLine += c;      // add it to the end of the currentLine
        }
      }
    }
    // Clear the header variable
    header = "";
    // Close the connection
    Serial.println("Client disconnected.");
    Serial.println("");
  }
  else{
  //  Serial.println("Client lost.");
    
  }
}
