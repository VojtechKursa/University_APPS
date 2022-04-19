#include "I2C_Radio.h"

RDS::RDS()
{
	Reset();
}



uint16_t RDS::GetPIcode()
{
	return piCode;
}

uint16_t RDS::GetAltFrequency()
{
	return alternativeFrequency;
}

uint8_t RDS::GetProgramType()
{
	return programType;
}

bool RDS::HasTrafficReports()
{
	return trafficReports;
}

const char* RDS::GetProgramService()
{
	return programService;
}

const char* RDS::GetRadioText()
{
	return radioText;
}



void RDS::Reset()
{
	this->piCode = 0;
	this->alternativeFrequency = 0;
	this->programType = 0;
	this->trafficReports = false;

	for (int i = 0; i < 9; i++)
	{
		this->programService[i] = 0;
	}

	for (int i = 0; i < 65; i++)
	{
		this->radioText[i] = 0;
	}
}

void RDS::Update(uint16_t block1, uint16_t block2, uint16_t block3, uint16_t block4, uint8_t blockErrorsByte)
{

	// Check for corrupted data
	bool blocksOK[4] = {true, true, true, true};
/*
	int mask = 0xC0;
	int value;
	for(int i = 3; i >= 0; i--)
	{
		value = (blockErrorsByte & mask) >> (2*i);

		if(value < 3)
			blocksOK[3 - i] = true;
		else
		{
			blocksOK[3 - i] = false;
			printf("WARN: Received RDS response with corrupted block %d, ignoring block.\n", 4 - i);
		}

		mask = mask >> 2;
	}
*/

	// Process blocks
	if(blocksOK[0])
	{
		piCode = block1;
	}

	if(blocksOK[1])
	{
		uint8_t groupType = (block2 & 0xF000) >> 12;
		bool groupB = (block2 & 0x0800) >> 11;

		if(!groupB)	// Group A
		{
			if(groupType == 0)
			{
				uint8_t segment = block2 & 0b11;

				if(blocksOK[2])
				{
					// Alternative frequency
				}
				if(blocksOK[3])
				{
					programService[segment * 2] = (block4 & 0xFF00) >> 8;
					programService[segment * 2 + 1] = block4 & 0xFF;
				}
			}
			else if(groupType == 2)
			{
				uint8_t segment = block2 & 0xF;

				if(blocksOK[2])
				{
					radioText[segment * 4] = (block3 & 0xFF00) >> 8;
					radioText[segment * 4 + 1] = block3 & 0xFF;
				}
				if(blocksOK[3])
				{
					radioText[segment * 4 + 2] = (block4 & 0xFF00) >> 8;
					radioText[segment * 4 + 3] = block4 & 0xFF;
				}
			}
		}
		else	// Group B
		{
			if(groupType == 0)
			{
				uint8_t segment = block2 & 0b11;

				if(blocksOK[2])
				{
					piCode = block3;
				}
				if(blocksOK[3])
				{
					programService[segment * 2] = (block4 & 0xFF00) >> 8;
					programService[segment * 2 + 1] = block4 & 0xFF;
				}
			}
		}
	}
}
