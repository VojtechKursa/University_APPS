/*
#include <I2C_LED.h>
 * I2CLED.cpp
 *
 *  Created on: Apr 12, 2022
 *      Author: kur0170
 */

#include "i2c-lib.h"
#include "i2c_dirs.h"

#include "I2C_LED.h"



I2C_LED::I2C_LED(int address)
{
	this->address = (0b0100 << 4) | ((address % 8) << 1);
}


bool I2C_LED::Bar(int length)
{
	uint8_t sentValue = 0;

	for(int i = 0; i < length; i++)
	{
		sentValue = (sentValue << 1) | 1;
	}

	int answers = 0;

	i2c_start();
	answers |= i2c_output(this->address | W);
	answers |= i2c_output(sentValue);
	i2c_stop();

	return answers == 0;
}

