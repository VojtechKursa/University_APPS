// **************************************************************************
//
//               Demo program for APPS labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 02/2022
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         Main program for LCD module
//
// **************************************************************************

#include "mbed.h"
#include "lcd_lib.h"
#include "graph_class.hpp"

DigitalOut g_led_PTA1( PTA1, 0 );
DigitalOut g_led_PTA2( PTA2, 0 );

DigitalIn g_but_PTC9( PTC9 );
DigitalIn g_but_PTC10( PTC10 );
DigitalIn g_but_PTC11( PTC11 );
DigitalIn g_but_PTC12( PTC12 );

int main()
{
	lcd_init();				// LCD initialization

	Point2D p1;
	p1.x = 10;
	p1.y = 20;

	RGB bg = {0,0,0};
	RGB fg = {255,255,0};

	Text t1(p1, "123456789ABCDabcd", fg, bg);
	t1.draw();

	p1.y += FONT_HEIGHT;
	Text t2(p1, "abcdefg", fg, bg, true);
	t2.draw();

	Point2D ul = {30, 40};
	Point2D lr = {200, 100};

	RectangleWithDiagonals r1(ul, lr, fg, bg);
	r1.draw();

	Point2D c_center = {150, 160};
	Circle c1(c_center, 50, fg, bg);
	c1.draw();

	return 0;
}
