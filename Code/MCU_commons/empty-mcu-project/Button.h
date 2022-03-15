#ifndef BUTTON_H_
#define BUTTON_H_


#include "mbed.h"


class Button
{
private:
	DigitalIn *button;
	int visibleState = 1;
	int lastState = 1;
	int currState = 1;

public:
	Button(DigitalIn *button);

	int Read();
	int ReadInverse();

	void Tick();
};


#endif /* BUTTON_H_ */
