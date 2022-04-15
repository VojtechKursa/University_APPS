#ifndef RDS_H_
#define RDS_H_

#include <stdio.h>

class RDS {
private:
	uint16_t piCode;
	uint16_t alternativeFrequency;
	uint8_t programType;
	bool trafficReports;
	char programService[9];
	char radioText[65];

public:
	RDS();

	uint16_t GetPIcode();
	uint16_t GetAltFrequency();
	uint8_t GetProgramType();
	bool HasTrafficReports();
	const char* GetProgramService();
	const char* GetRadioText();

	void Reset();
	void Update(uint16_t block1, uint16_t block2, uint16_t block3, uint16_t block4, uint8_t blockErrorsByte);
};

#endif /* RDS_H_ */
