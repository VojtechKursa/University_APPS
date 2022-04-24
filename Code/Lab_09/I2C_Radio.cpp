/*
 * I2C_Radio.cpp
 *
 *  Created on: Apr 12, 2022
 *      Author: kur0170
 */

#include <I2C_Radio.h>



I2C_Radio::I2C_Radio(void)
{
	int ack = si4735_init();

	if (ack != 0)
		printf("FAIL: Initialization of SI4735 finished with error (%d).\n", ack);
	else
	{
		printf("SUCCESS: SI4735 initialized.\n");
	}
}



void I2C_Radio::UpdateStatus()
{

}

void I2C_Radio::UpdateSignalQuality()
{

}



// volume range: 0 - 63
bool I2C_Radio::SetVolume(int volume)
{
	uint8_t correctedVolume = volume % 63;
	int check = 0;

	i2c_start();
	check |= i2c_output(this->address | W);
	check |= i2c_output(0x12);
	check |= i2c_output(0x00);
	check |= i2c_output(0x40);
	check |= i2c_output(0x00);
	check |= i2c_output(correctedVolume);
	i2c_stop();


	if(check)
	{
		printf("FAIL: Set volume to %d failed.\n", correctedVolume);
		return false;
	}
	else
	{
		this->volume = correctedVolume;
		printf("SUCCESS: Volume set to %d.\n", correctedVolume);
		return true;
	}
}

bool I2C_Radio::ChangeVolume(int change)
{
	int newVolume = volume + change;

	if(SetVolume(newVolume))
	{
		volume = newVolume;
		return true;
	}
	else
		return false;
}

// frequency = desired frequency in MHz * 100
bool I2C_Radio::SetFrequency(int frequency)
{
	int check = 0;
	frequency *= 100;
	uint8_t freqHi = (frequency & 0xFF00 >> 8);
	uint8_t freqLo = frequency & 0xFF;

	i2c_start();
	check |= i2c_output(this->address | W);
	check |= i2c_output(0x20);
	check |= i2c_output(0x00);
	check |= i2c_output(freqHi);
	check |= i2c_output(freqLo);
	check |= i2c_output(0x00);
	i2c_stop();

	if(check)
	{
		printf("FAIL: Set frequency to %d failed.\n", frequency);
		return false;
	}
	else
	{
		printf("SUCCESS: Frequency set to %d.\n", frequency);
		return true;
	}
}

bool I2C_Radio::AutoTune(bool up)
{
	int check = 0;
	uint8_t parameterByte = (up ? 0x0 : 0x8) | 0x4;

	i2c_start();
	check |= i2c_output(this->address | W);
	check |= i2c_output(0x21);
	check |= i2c_output(parameterByte);
	i2c_stop();

	return check == 0;
}



int I2C_Radio::GetVolume()
{
	return volume;
}

int I2C_Radio::GetFrequency()
{
	return frequency;
}

bool I2C_Radio::IsStereo()
{
	return isStereo;
}

bool I2C_Radio::IsTuned()
{
	return isTuned;
}



void I2C_Radio::Print()
{

}
