// **************************************************************************
#include <I2C_LED.h>
//
//               Demo program for APPS labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 02/2022
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         Main program for I2C bus
//
// **************************************************************************

#include <mbed.h>
#include <stdio.h>

#include "i2c-lib.h"
#include "si4735-lib.h"

#include "i2c_dirs.h"

#include "I2C_LED.h"
#include "I2C_Radio.h"


//************************************************************************


#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

DigitalOut g_led_PTA1( PTA1, 0 );
DigitalOut g_led_PTA2( PTA2, 0 );

DigitalIn g_but_PTC9( PTC9 );
DigitalIn g_but_PTC10( PTC10 );
DigitalIn g_but_PTC11( PTC11 );
DigitalIn g_but_PTC12( PTC12 );

I2C_Radio radio;

void buttonCheck()
{
	if(g_but_PTC9.read() == 0 && g_but_PTC10.read() == 0)
	{
		printf("Autotune triggered\n.");
		radio.AutoTune(true);
	}
	else
	{
		if(g_but_PTC12.read() == 0)
			radio.ChangeVolume(2);
		if(g_but_PTC11.read() == 0)
			radio.ChangeVolume(-2);
		if(g_but_PTC10.read() == 0)
			radio.SetFrequency(radio.GetFrequency() + 50);
		if(g_but_PTC9.read() == 0)
			radio.SetFrequency(radio.GetFrequency() - 50);
	}
}

int main(void)
{
	i2c_init();

	I2C_LED leds(2);
	printf("%d\n", leds.Bar(6));

	radio = I2C_Radio();
	radio.SetFrequency(10140);
	radio.SetVolume(40);

	Ticker buttonTicker;
	buttonTicker.attach(buttonCheck, 200ms);

	while(true)
	{}

	return 0;
}