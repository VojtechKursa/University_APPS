#ifndef I2CLED_H_
#define I2CLED_H_

#include <stdint.h>

class I2C_LED {
private:
	uint8_t address;

public:
	I2C_LED(int address);
	
	bool Set(uint8_t value);
	uint8_t Get();

	bool Bar(int length);
};

#endif /* I2C_LED_H_ */
