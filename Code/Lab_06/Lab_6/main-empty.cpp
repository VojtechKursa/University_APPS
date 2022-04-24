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

#define rgbLedsLength 3
RGBLED *rgbLeds[rgbLedsLength];

#define buttonsLength 4
Button *buttons[buttonsLength];
DigitalIn *buttonsRaw[buttonsLength];


void ledTickerCallback()
{
	static int step = 0;

	for(int i = 0; i < ledsLength; i++)
	{
		leds[i]->Tick(step);
	}

	for(int i = 0; i < rgbLedsLength; i++)
	{
		rgbLeds[i]->Tick(step);
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


	RGBLED* rgb0 = new RGBLED(new LED(new DigitalOut(PTB2)), new LED(new DigitalOut(PTB3)), new LED(new DigitalOut(PTB9)));
	RGBLED* rgb1 = new RGBLED(new LED(new DigitalOut(PTB10)), new LED(new DigitalOut(PTB11)), new LED(new DigitalOut(PTB18)));
	RGBLED* rgb2 = new RGBLED(new LED(new DigitalOut(PTB19)), new LED(new DigitalOut(PTB20)), new LED(new DigitalOut(PTB23)));

	rgbLeds[0] = rgb0;
	rgbLeds[1] = rgb1;
	rgbLeds[2] = rgb2;


	DigitalIn *buttonRaw1 = new DigitalIn(PTC9);
	DigitalIn *buttonRaw2 = new DigitalIn(PTC10);
	DigitalIn *buttonRaw3 = new DigitalIn(PTC11);
	DigitalIn *buttonRaw4 = new DigitalIn(PTC12);

	buttonsRaw[0] = buttonRaw1;
	buttonsRaw[1] = buttonRaw2;
	buttonsRaw[2] = buttonRaw3;
	buttonsRaw[3] = buttonRaw4;

	for(int i = 0; i < 4; i++)
	{
		buttons[i] = new Button(buttonsRaw[i]);
	}
}

void resetRgbLeds()
{
	rgbLeds[0]->SetColor(255, 0, 0);
	rgbLeds[1]->SetColor(0, 255, 0);
	rgbLeds[2]->SetColor(0, 0, 255);
}

void buttonActions()
{
	static int currRedLed = 0;
	static int currGreenLed = 1;
	static int currBlueLed = 2;

	if(!buttonsRaw[0]->read())
	{
		RGBLED* currLed = rgbLeds[currRedLed];
		RGBLED* nextLed = rgbLeds[(currRedLed + 1) % rgbLedsLength];

		currLed->SetR(currLed->GetR() - 15);
		nextLed->SetR(nextLed->GetR() + 15);

		if(currLed->GetR() == 0)
			currRedLed = (currRedLed + 1) % 3;
	}

	if(!buttonsRaw[1]->read())
	{
		RGBLED* currLed = rgbLeds[currGreenLed];
		RGBLED* nextLed = rgbLeds[(currGreenLed + 1) % rgbLedsLength];

		currLed->SetG(currLed->GetG() - 15);
		nextLed->SetG(nextLed->GetG() + 15);

		if(currLed->GetG() == 0)
			currGreenLed = (currGreenLed + 1) % 3;
	}

	if(!buttonsRaw[2]->read())
	{
		RGBLED* currLed = rgbLeds[currBlueLed];
		RGBLED* nextLed = rgbLeds[(currBlueLed + 1) % rgbLedsLength];

		currLed->SetB(currLed->GetB() - 15);
		nextLed->SetB(nextLed->GetB() + 15);

		if(currLed->GetB() == 0)
			currBlueLed = (currBlueLed + 1) % 3;
	}

	if(!buttonsRaw[3]->read())
	{
		resetRgbLeds();
	}
}

int main()
{
	fillArrays();

	for(int i = 5; i <= 40; i += 5)
	{
		leds[i / 5]->SetBrightnessPercent(i);
	}

	resetRgbLeds();

	Ticker ledTicker;
	ledTicker.attach(ledTickerCallback, 80us);

	Ticker buttonTicker;
	buttonTicker.attach(buttonTickerCallback, 50ms);

	while(true)
	{ }

	return 0;
}
