// **************************************************************************
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

#include <stdio.h>

#include <mbed.h>

#include "i2c-lib.h"
#include "si4735-lib.h"

#include "i2c_dirs.h"

#include "I2C_LED.h"
#include "I2C_Radio.h"
#include "Button.h"


//************************************************************************


#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

Button* buttons[4];

I2C_Radio radio;

void radioUpdate()
{
	radio.Update();
}

void buttonCheck()
{
	for(int i = 0; i < 4; i++)
	{
		buttons[i]->Tick();
	}

	if(buttons[0]->WasPressed() && buttons[1]->WasPressed())
	{
		buttons[0]->ClearPressed();
		buttons[1]->ClearPressed();
		radio.Print();
	}

	if(buttons[0]->WasPressed())
	{
		buttons[0]->ClearPressed();
		radio.PrevChannel();
	}
	if(buttons[1]->WasPressed())
	{
		buttons[1]->ClearPressed();
		radio.NextChannel();
	}
	if(buttons[2]->WasPressed())
	{
		buttons[2]->ClearPressed();
		radio.ChangeVolume(-2);
	}
	if(buttons[3]->WasPressed())
	{
		buttons[3]->ClearPressed();
		radio.ChangeVolume(2);
	}

	if(buttons[0]->WasHeld())
	{
		buttons[0]->ClearHeld();
		radio.AutoTune(false);
	}
	if(buttons[1]->WasHeld())
	{
		buttons[1]->ClearHeld();
		radio.AutoTune(true);
	}
}

void InitControls()
{
	DigitalOut led_PTA1( PTA1, 0 );
	DigitalOut led_PTA2( PTA2, 0 );

	buttons[0] = new Button(new DigitalIn(PTC12));
	buttons[1] = new Button(new DigitalIn(PTC9));
	buttons[2] = new Button(new DigitalIn(PTC10));
	buttons[3] = new Button(new DigitalIn(PTC11));
}

void InitRadio()
{
	int ack = si4735_init();

	if (ack != 0)
		printf("FAIL: Initialization of SI4735 finished with error (%d).\n", ack);
	else
		printf("SUCCESS: SI4735 initialized.\n");

	wait_us(100000);

	radio.SetLEDs(new I2C_LED(2));

	radio.SetFrequency(10140);
	//radio.SetVolume(30);
	radio.Update();
	radio.Print();
}

void Init()
{
	i2c_init();
	InitControls();
	InitRadio();

	printf("Init complete.\n");
}

void CleanUp()
{
	for(int i = 0; i < 4; i++)
	{
		delete buttons[i];
		buttons[i] = nullptr;
	}
}

int main(void)
{
	Init();

	Ticker buttonTicker, radioTicker;
	buttonTicker.attach(buttonCheck, 100ms);
	radioTicker.attach(radioUpdate, 1s);

	while(true)
	{}

	CleanUp();

	return 0;
}

/*
int main( void )
{
	uint8_t l_S1, l_S2, l_RSSI, l_SNR, l_MULT, l_CAP;
	uint8_t l_ack = 0;

	printf( "K64F-KIT ready...\r\n" );

	i2c_init();
655,35
	// communication with 8 bit expander PCF8574

	// start communication
	i2c_start();

	// PCF8574 addressing
	// The address is composed from 3 parts!
	//l_ack = i2c_output( HWADR_PCF8574 | A012 | W );

	// Check l_ack! Return value must be 0!
	// ....

	//l_ack = i2c_output( Any_8_bit_value );
	// selected LEDs should light

	// stop communication
	i2c_stop();

	if ( ( l_ack = si4735_init() ) != 0 )
	{
		printf( "Initialization of SI4735 finish with error (%d)\r\n", l_ack );
		return 0;
	}
	else
		printf( "SI4735 initialized.\r\n" );

	printf( "\nTunig of radio station...\r\n" );

	// Required frequency in MHz * 100
	int l_freq = 10140; // Radiozurnal

	// Tuning of radio station
	i2c_start();
	l_ack |= i2c_output( SI4735_ADDRESS | W);
	l_ack |= i2c_output( 0x20 );			// FM_TUNE_FREQ
	l_ack |= i2c_output( 0x00 );			// ARG1
	l_ack |= i2c_output( l_freq >> 8 );		// ARG2 - FreqHi
	l_ack |= i2c_output( l_freq & 0xff );	// ARG3 - FreqLo
	l_ack |= i2c_output( 0x00 );			// ARG4
	i2c_stop();
	// Check l_ack!
	// if...

	// Tuning process inside SI4735
	wait_us( 100000 );
	printf( "... station tuned.\r\n\n" );

	// Example of reading of tuned frequency
	i2c_start();
	l_ack |= i2c_output( SI4735_ADDRESS | W );
	l_ack |= i2c_output( 0x22 );			// FM_TUNE_STATUS
	l_ack |= i2c_output( 0x00 );			// ARG1
	// repeated start
	i2c_start();
	// change direction of communication
	l_ack |= i2c_output( SI4735_ADDRESS | R );
	// read data
	l_S1 = i2c_input();
	i2c_ack();
	l_S2 = i2c_input();
	i2c_ack();
	l_freq = ( uint32_t ) i2c_input() << 8;
	i2c_ack();
	l_freq |= i2c_input();
	i2c_ack();
	l_RSSI = i2c_input();
	i2c_ack();
	l_SNR = i2c_input();
	i2c_ack();
	l_MULT = i2c_input();
	i2c_ack();
	l_CAP = i2c_input();
	i2c_nack();
	i2c_stop();

	if ( l_ack != 0 )
		printf( "Communication error!\r\n" );
	else
		printf( "Current tuned frequency: %d.%dMHz\r\n", l_freq / 100, l_freq % 100 );

	return 0;
}
*/
