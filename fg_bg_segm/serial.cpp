#include "serial.h"

void serial_setup ( void ) {
    ardu.Open(PORT);
    ardu.SetBaudRate(BAUDRATE);
    ardu.SetCharSize(SerialStreamBuf::CHAR_SIZE_8);
}

