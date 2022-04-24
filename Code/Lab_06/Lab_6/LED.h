#ifndef LED_H_
#define LED_H_


#include "mbed.h"


class LED
{
private:
	DigitalOut *led;
	int currValue;
	int nextValue = 0;
	int brightness = 0;

public:
	LED(DigitalOut *led);
	LED(DigitalOut *led, int brightness);

	int GetBrightness();
	int GetBrightnessPercent();

	void SetBrightness(int value);
	void SetBrightnessPercent(int percent);

	void Write(int value);
	int Read();

	void Tick(int step);
};


#endif /* LED_H_ */
