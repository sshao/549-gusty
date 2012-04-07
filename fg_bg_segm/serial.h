#ifndef _SERIAL_H_
#define _SERIAL_H_

#include <SerialStream.h>

#define PORT "/dev/ttyACM0"
#define BAUDRATE (SerialStreamBuf::BAUD_9600)

using namespace LibSerial;

// Serial stream to arduino
extern SerialStream ardu;

// Set up serial communication with the Arduino (open ports, set baud rate,
// set char size)
void serial_setup(void);

#endif
