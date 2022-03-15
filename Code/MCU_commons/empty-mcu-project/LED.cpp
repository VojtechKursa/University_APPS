#include "LED.h"

LED::LED(DigitalOut *led)
{
	this->led = led;
	this->currValue = led->read();
}

LED::LED(DigitalOut *led, int brightness)
{
	this->led = led;
	this->currValue = led->read();
	this->brightness = brightness;
}


int LED::GetBrightness()
{
	return brightness;
}

int LED::GetBrightnessPercent()
{
	return (brightness * 100) / 255;
}


void LED::SetBrightness(int value)
{
	if(value >= 0 && value <= 255)
		brightness = value;
}

void LED::SetBrightnessPercent(int percent)
{
	SetBrightness((percent * 255) / 100);
}


void LED::Write(int value)
{
	led->write(value);
	currValue = value;
}

int LED::Read()
{
	return led->read();
}


void LED::Tick(int step)
{
	nextValue = step <= brightness ? 1 : 0;

	if (nextValue != currValue)
	{
		currValue = nextValue;
		led->write(nextValue);
	}
}
