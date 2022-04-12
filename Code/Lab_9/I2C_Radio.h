/*
 * I2C_Radio.h
 *
 *  Created on: Apr 12, 2022
 *      Author: kur0170
 */

#ifndef I2C_RADIO_H_
#define I2C_RADIO_H_

#include "si4735-lib.h"
#include "i2c-lib.h"
#include "i2c_dirs.h"

#include <stdint.h>

class I2C_Radio {
private:
	uint8_t address = 0x22;
	int volume = 0;
	int frequency = 0;
	bool isTuned = false;
	bool isStereo = false;
	int SNR;

	void UpdateStatus();
	void UpdateSignalQuality();

public:
	I2C_Radio();

	bool SetVolume(int volume);
	bool ChangeVolume(int change);
	bool SetFrequency(int frequency);
	bool AutoTune(bool up);

	int GetVolume();
	int GetFrequency();
	bool IsStereo();
	bool IsTuned();

	void Print();
};

#endif /* I2C_RADIO_H_ */
