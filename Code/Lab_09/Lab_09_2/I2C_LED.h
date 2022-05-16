/*
 * I2CLED.h
 *
 *  Created on: Apr 12, 2022
 *      Author: kur0170
 */

#ifndef I2CLED_H_
#define I2CLED_H_

#include <stdint.h>

class I2C_LED {
private:
	uint8_t address;

public:
	I2C_LED(int address);

	bool Bar(int length);
};

#endif /* I2C_LED_H_ */
