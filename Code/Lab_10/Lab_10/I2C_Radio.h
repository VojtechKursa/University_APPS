#ifndef I2C_RADIO_H_
#define I2C_RADIO_H_

#include <stdint.h>
#include <string>

#include "si4735-lib.h"
#include "i2c-lib.h"
#include "i2c_dirs.h"

#include "RDS.h"
#include "I2C_LED.h"

using namespace std;

class I2C_Radio {
private:
	uint8_t address = 0x22;

	I2C_LED* leds = nullptr;

	uint8_t currChannel = 0;
	uint16_t channelList[15] = {
			8760,
			8900,
			8980,
			9040,
			9100,
			9180,
			9250,
			9280,
			9370,
			9470,
			9640,
			9770,
			9900,
			10050,
			10140
	};

	uint8_t volume = 0;
	uint16_t frequency = 0;
	bool isTuned = false;
	bool isValid = false;
	bool isStereo = false;
	bool isMuted = false;
	uint8_t SNR = 0;
	uint8_t RSSI = 0;
	RDS rds;

	bool ProcessRDSQueue();
	int ProcessRDSBlock();
	bool UpdateTuneStatus();
	bool UpdateSignalQualityStatus();
	void GetResponse(uint8_t* buffer, int bufferStart, int responseLength);

public:
	uint8_t GetVolume();
	uint16_t GetFrequency();
	bool IsStereo();
	bool IsTuned();
	bool IsValid();
	bool IsMuted();
	uint8_t GetRSSI();
	uint8_t GetSNR();

	uint16_t GetPIcode();
	uint16_t GetAltFrequency();
	uint8_t GetProgramType();
	bool HasTrafficReports();
	const char* GetStationName();
	const char* GetRadioText();

	void SetLEDs(I2C_LED* leds);
	bool SetVolume(int volume);
	bool ChangeVolume(int change);
	bool SetFrequency(int frequency);
	bool NextChannel();
	bool PrevChannel();
	
	bool AutoTune(bool up);
	void Update();

	void Print();
};

#endif /* I2C_RADIO_H_ */
