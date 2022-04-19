#ifndef BUTTON_H_
#define BUTTON_H_


#include "mbed.h"


class Button
{
private:
	DigitalIn* button;

	bool wasPressed = false;
	bool wasHeld = false;

public:
	Button(DigitalIn* button);
	~Button();

	bool WasPressed();
	bool WasHeld();

	void ClearPressed();
	void ClearHeld();

	void Tick();
};


#endif /* BUTTON_H_ */
