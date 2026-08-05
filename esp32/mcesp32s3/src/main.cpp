#include <Arduino.h>
#include "tokanizer.h"


void setup()
{
    Serial.begin(115200);
    delay(2000);
    String command =
        "switch on light";
    uint8_t tokens[6];
    tokenize(
        command,
        tokens,
        6
    );
    Serial.println("Tokens:");
    for(int i=0;i<6;i++)
    {
        Serial.println(tokens[i]);
    }
}


void loop()
{

}