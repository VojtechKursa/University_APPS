/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "Button.h"
#include "LED.h"
#include "RGBLED.h"


#define ledsLength 2
LED *leds[ledsLength];

#define buttonsLength 4
Button *buttons[buttonsLength];


void ledTickerCallback()
{
	static int step = 0;

	for(int i = 0; i < ledsLength; i++)
	{
		leds[i]->Tick(step);
	}

	step = (step + 1) % 256;
}

void buttonActions();
void buttonTickerCallback()
{
	for(int i = 0; i < buttonsLength; i++)
	{
		buttons[i]->Tick();
	}

	buttonActions();
}

void fillArrays()
{
	LED *led1 = new LED(new DigitalOut(PTA1), 25);
	LED *led2 = new LED(new DigitalOut(PTA2), 50);

	leds[0] = led1;
	leds[1] = led2;

	Button *button1 = new Button(new DigitalIn(PTC9));
	Button *button2 = new Button(new DigitalIn(PTC10));
	Button *button3 = new Button(new DigitalIn(PTC11));
	Button *button4 = new Button(new DigitalIn(PTC12));

	buttons[0] = button1;
	buttons[1] = button2;
	buttons[2] = button3;
	buttons[3] = button4;
}


void buttonActions()
{
	if(buttons[0]->ReadInverse())
		leds[0]->SetBrightnessPercent(leds[0]->GetBrightnessPercent() - 10);
	if(buttons[1]->ReadInverse())
		leds[0]->SetBrightnessPercent(leds[0]->GetBrightnessPercent() + 10);
	if(buttons[2]->ReadInverse())
		leds[1]->SetBrightnessPercent(leds[1]->GetBrightnessPercent() - 10);
	if(buttons[3]->ReadInverse())
		leds[1]->SetBrightnessPercent(leds[1]->GetBrightnessPercent() + 10);
}

int main()
{
	fillArrays();

	Ticker ledTicker;
	ledTicker.attach(ledTickerCallback, 40us);

	Ticker buttonTicker;
	buttonTicker.attach(buttonTickerCallback, 50ms);

	while(true)
	{ }

	return 0;
}
