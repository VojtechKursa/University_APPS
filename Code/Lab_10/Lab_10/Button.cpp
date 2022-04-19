#include "Button.h"

Button::Button(DigitalIn* button)
{
	this->button = button;
}

Button::~Button()
{
	delete this->button;
}



bool Button::WasPressed()
{
	return wasPressed;
}

bool Button::WasHeld()
{
	return wasHeld;
}

void Button::ClearPressed()
{
	wasPressed = false;
}

void Button::ClearHeld()
{
	wasHeld = false;
}



void Button::Tick()
{
	static int heldFor = 0;

	if(!button->read())
	{
		heldFor++;
	}
	else
	{
		if(heldFor > 0)
		{
			if(heldFor >= 5)
				wasHeld = true;
			else
				wasPressed = true;

			heldFor = 0;
		}
	}
}
