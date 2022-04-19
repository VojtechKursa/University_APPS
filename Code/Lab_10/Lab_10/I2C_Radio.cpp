#include <I2C_Radio.h>



bool I2C_Radio::ProcessRDSQueue()
{
	int remainingQueue;

	do
	{
		remainingQueue = ProcessRDSBlock();
	} while(remainingQueue > 1);

	return remainingQueue != -1;
}

// Returns: -1 if error, otherwise length of RDS Queue (0 means no data in Queue so no data were processed, 1 means this was the last block in Queue)
int I2C_Radio::ProcessRDSBlock()
{
	int acks = 0;
	uint8_t response[13];

	// Send request
	i2c_start();
	acks |= i2c_output(this->address | W);
	acks |= i2c_output(0x24);
	acks |= i2c_output(0x00);

	// Receive data
	i2c_start();
	acks |= i2c_output(this->address | R);
	GetResponse(response, 0, 13);
	i2c_stop();

	// Check for errors
	if(acks)
	{
		printf("FAIL: Received NACK while requesting RDS update.\n");
		return -1;
	}
	if(!(response[2] & 0b1))
	{
		printf("WARN: Received unsynchronized RDS response, ignoring response.\n");
		return -1;
	}
	if(response[3] == 0)
	{
		printf("WARN: Received RDS response with empty RDS Queue, ignoring response.\n");
		return 0;
	}

	uint16_t block1 = (response[4] << 8) | response[5];
	uint16_t block2 = (response[6] << 8) | response[7];
	uint16_t block3 = (response[8] << 8) | response[9];
	uint16_t block4 = (response[10] << 8) | response[11];
	uint8_t blockErrorsByte = response[12];

	rds.Update(block1, block2, block3, block4, blockErrorsByte);

	return response[3];
}

bool I2C_Radio::UpdateTuneStatus()
{
	int ack = 0;
	uint8_t response[8];

	i2c_start();
	
	ack |= i2c_output(this->address | W);
	ack |= i2c_output(0x22);
	ack |= i2c_output(0x00);
	
	i2c_start();
	ack |= i2c_output(this->address | R);
	GetResponse(response, 0, 8);

	i2c_stop();

	if(ack == 0)
	{
		this->isTuned = response[0] & 0b1;
		this->isValid = response[1] & 0b1;
		this->frequency = (response[2] << 8) | response[3];
		this->RSSI = response[4];
		this->SNR = response[5];
	}
	else
		printf("FAIL: Returned NACK during UpdateTuneStatus.\n");

	return ack == 0;
}

bool I2C_Radio::UpdateSignalQualityStatus()
{
	int ack = 0;
	uint8_t response[8];

	i2c_start();
	
	ack |= i2c_output(this->address | W);
	ack |= i2c_output(0x23);
	ack |= i2c_output(0x00);

	i2c_start();
	ack |= i2c_output(this->address | R);
	GetResponse(response, 0, 8);

	i2c_stop();

	if(ack == 0)
	{
		this->isTuned = response[0] & 0b1;
		this->isValid = response[2] & 0b1;
		this->isMuted = response[2] & 0x8;
		this->isStereo = response[3] & 0x80;
		this->RSSI = response[4];
		this->SNR = response[5];
	}
	else
		printf("FAIL: Returned NACK during UpdateSignalQualityStatus.\n");
	return ack == 0;
}

void I2C_Radio::GetResponse(uint8_t* buffer, int bufferStart, int responseLength)
{
	for(int i = 0; i < responseLength; i++)
	{
		buffer[bufferStart + i] = i2c_input();

		if(i == responseLength - 1)
			i2c_nack();
		else
			i2c_ack();
	}
}



uint8_t I2C_Radio::GetVolume()
{
	return volume;
}

uint16_t I2C_Radio::GetFrequency()
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

bool I2C_Radio::IsValid()
{
	return isValid;
}

bool I2C_Radio::IsMuted()
{
	return isMuted;
}

uint8_t I2C_Radio::GetSNR()
{
	return SNR;
}

uint8_t I2C_Radio::GetRSSI()
{
	return RSSI;
}

uint16_t I2C_Radio::GetPIcode()
{
	return rds.GetPIcode();
}

uint16_t I2C_Radio::GetAltFrequency()
{
	return rds.GetAltFrequency();
}

uint8_t I2C_Radio::GetProgramType()
{
	return rds.GetProgramType();
}

bool I2C_Radio::HasTrafficReports()
{
	return rds.HasTrafficReports();
}

const char* I2C_Radio::GetStationName()
{
	return rds.GetProgramService();
}

const char* I2C_Radio::GetRadioText()
{
	return rds.GetRadioText();
}



void I2C_Radio::SetLEDs(I2C_LED* leds)
{
	this->leds = leds;
}

// volume range: 0 - 63
bool I2C_Radio::SetVolume(int volume)
{
	uint8_t correctedVolume = volume > 63 ? 63 : (volume < 0 ? 0 : volume);
	int check = 0;

	i2c_start();
	check |= i2c_output(this->address | W);
	check |= i2c_output(0x12);
	check |= i2c_output(0x00);
	check |= i2c_output(0x40);
	check |= i2c_output(0x00);
	check |= i2c_output(correctedVolume);
	i2c_stop();

	wait_us( 100000 );

	if(check)
	{
		printf("FAIL: Set volume to %d (corrected from %d) failed.\n", correctedVolume, volume);
		return false;
	}
	else
	{
		this->volume = correctedVolume;
		printf("SUCCESS: Volume set to %d (corrected from %d).\n", correctedVolume, volume);
		return true;
	}
}

bool I2C_Radio::ChangeVolume(int change)
{
	int newVolume = this->volume + change;

	return SetVolume(newVolume);
}

// frequency = desired frequency in MHz * 100
bool I2C_Radio::SetFrequency(int frequency)
{
	int check = 0;

	i2c_start();
	check |= i2c_output(this->address | W);
	check |= i2c_output(0x20);
	check |= i2c_output(0x00);
	check |= i2c_output(frequency >> 8);
	check |= i2c_output(frequency & 0xFF);
	check |= i2c_output(0x00);
	i2c_stop();


	wait_us( 100000 );

	if(check)
	{
		printf("FAIL: Set frequency to %d failed.\n", frequency);
		return false;
	}
	else
	{
		printf("SUCCESS: Frequency set to %d (0x%x 0x%x).\n", frequency, frequency >> 8, frequency & 0xFF);
		this->frequency = frequency;
		rds.Reset();
		return true;
	}
}

bool I2C_Radio::NextChannel()
{
	currChannel = (currChannel + 1) % 15;

	return SetFrequency(channelList[currChannel]);
}

bool I2C_Radio::PrevChannel()
{
	if(currChannel == 0)
		currChannel = 14;
	else
		currChannel = (currChannel - 1) % 15;

	return SetFrequency(channelList[currChannel]);
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

	if(check)
	{
		printf("FAIL: Autotune request failed.\n");
		return false;
	}
	else
	{
		printf("SUCCESS: Autotune request accepted. Waiting 250ms... ");

		wait_us(250000);
		printf("Complete.\n");

		rds.Reset();
		UpdateTuneStatus();

		return true;
	}
}

void I2C_Radio::Update()
{
	UpdateTuneStatus();
	UpdateSignalQualityStatus();
	//ProcessRDSQueue();

	leds->Bar(SNR);
}



void I2C_Radio::Print()
{
	printf("Radio status:\n");
	printf("\tVolume: %d\n", volume);
	printf("\tFrequency: %d,%d MHz\n", frequency / 100, frequency % 100);
	printf("\tIsTuned/IsValid: %d/%d\n", isTuned, isValid);
	printf("\tIsMuted/IsStereo: %d/%d\n", isMuted, isStereo);
	printf("\tSNR: %d\n", SNR);
	printf("\tRSSI: %d\n", RSSI);
	printf("\tRDS:\n");
	printf("\t\tPI code: %x\n", GetPIcode());
	printf("\t\tStation name: %s\n", GetStationName());
	printf("\t\tProgram type: %d\n", GetProgramType());
	printf("\t\tRadiotext: %s\n", GetRadioText());
	//printf("\t\tAlternative frequency: %d,%d MHz\n", rds.GetAltFrequency() / 100, rds.GetAltFrequency() % 100);
	printf("\n");
}
