/* mbed Microcontroller Library
 * Copyright (c) 2019 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 */

#include "mbed.h"
#include "Button.h"
#include "LED.h"
#include "RGBLED.h"


#define ledsLength 8
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
	LED *led1 = new LED(new DigitalOut(PTC0));
	LED *led2 = new LED(new DigitalOut(PTC1));
	LED *led3 = new LED(new DigitalOut(PTC2));
	LED *led4 = new LED(new DigitalOut(PTC3));
	LED *led5 = new LED(new DigitalOut(PTC4));
	LED *led6 = new LED(new DigitalOut(PTC5));
	LED *led7 = new LED(new DigitalOut(PTC7));
	LED *led8 = new LED(new DigitalOut(PTC8));

	leds[0] = led1;
	leds[1] = led2;
	leds[2] = led3;
	leds[3] = led4;
	leds[4] = led5;
	leds[5] = led6;
	leds[6] = led7;
	leds[7] = led8;

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

	for(int i = 5; i <= 40; i += 5)
	{
		leds[i / 5]->SetBrightnessPercent(i);
	}

	Ticker ledTicker;
	ledTicker.attach(ledTickerCallback, 40us);

	Ticker buttonTicker;
	buttonTicker.attach(buttonTickerCallback, 50ms);

	while(true)
	{ }

	return 0;
}
