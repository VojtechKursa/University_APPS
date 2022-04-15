#include "i2c-lib.h"
#include "i2c_dirs.h"

#include "I2C_LED.h"



I2C_LED::I2C_LED(int address)
{
	this->address = (0b0100 << 4) | ((address % 8) << 1);
}



bool I2C_LED::Set(uint8_t value)
{
	int answers = 0;

	i2c_start();
	answers |= i2c_output(this->address | W);
	answers |= i2c_output(value);
	i2c_stop();

	return answers == 0;
}

uint8_t I2C_LED::Get()
{
	i2c_start();
	i2c_output(this->address | R);

	uint8_t value = i2c_input();
	i2c_ack();

	i2c_stop();

	return value;
}



bool I2C_LED::Bar(int length)
{
	uint8_t sentValue = 0;

	for(int i = 0; i < length; i++)
	{
		sentValue = (sentValue << 1) | 1;
	}

	return Set(sentValue);
}