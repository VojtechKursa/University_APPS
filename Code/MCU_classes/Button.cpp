#include "Button.h"

Button::Button(DigitalIn *button)
{
	this->button = button;
}

int Button::Read()
{
	int returnValue = visibleState;
	visibleState = 1;
	return returnValue;
}

int Button::ReadInverse()
{
	return !Read();
}

void Button::Tick()
{
	currState = button->read();

	if(currState != lastState)
	{
		visibleState = lastState = currState;
	}
}
