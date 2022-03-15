#ifndef RGBLED_H_
#define RGBLED_H_


#include "LED.h"


class RGBLED
{
private:
	LED *LedR, *LedG, *LedB;

public:
	RGBLED(LED *LedR, LED *LedG, LED *LedB);

	void SetR(int value);
	void SetG(int value);
	void SetB(int value);

	int GetR();
	int GetG();
	int GetB();

	void SetColor(int r, int g, int b);

	void Tick(int step);
};


#endif /* RGBLED_H_ */
