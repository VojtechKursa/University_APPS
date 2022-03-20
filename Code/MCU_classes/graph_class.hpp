// **************************************************************************
//
//               Demo program for labs
//
// Subject:      Computer Architectures and Parallel systems
// Author:       Petr Olivka, petr.olivka@vsb.cz, 09/2019
// Organization: Department of Computer Science, FEECS,
//               VSB-Technical University of Ostrava, CZ
//
// File:         OpenCV simulator of LCD
//
// **************************************************************************

#include "lcd_lib.h"

#include "font8x8.h"
#include <math.h>

// Simple graphic interface

struct Point2D 
{
    int32_t x, y;
};

struct RGB
{
    uint8_t r, g, b;
};

class GraphElement
{
public:
    // foreground and background color
    RGB m_fg_color, m_bg_color;

    // constructor
    GraphElement( RGB t_fg_color, RGB t_bg_color ) : 
        m_fg_color( t_fg_color ), m_bg_color( t_bg_color ) {}

    // ONLY ONE INTERFACE WITH LCD HARDWARE!!!
    void drawPixel( int32_t t_x, int32_t t_y ) { lcd_put_pixel( t_x, t_y, convert_RGB888_to_RGB565( m_fg_color ) ); }
    
    // Draw graphics element
    virtual void draw() = 0;
    
    // Hide graphics element
    virtual void hide() { swap_fg_bg_color(); draw(); swap_fg_bg_color(); }


private:
    // swap foreground and backgroud colors
    void swap_fg_bg_color() { RGB l_tmp = m_fg_color; m_fg_color = m_bg_color; m_bg_color = l_tmp; } 

    // IMPLEMENT!!!
    // conversion of 24-bit RGB color into 16-bit color format
    uint16_t convert_RGB888_to_RGB565( RGB t_color )
    {
    	return (uint16_t)(((t_color.r & 0xF8) << 8) | ((t_color.g & 0xFC) << 3) | ((t_color.b & 0xF8) >> 3));
    }
};


class Pixel : public GraphElement
{
public:
    // constructor
    Pixel( Point2D t_pos, RGB t_fg_color, RGB t_bg_color ) : GraphElement( t_fg_color, t_bg_color ), m_pos( t_pos ) {}
    // Draw method implementation
    virtual void draw() { drawPixel( m_pos.x, m_pos.y ); }
    // Position of Pixel
    Point2D m_pos;
};


class Circle : public GraphElement
{
public:
    // Center of circle
    Point2D m_center;
    // Radius of circle
    int32_t radius;

    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) : 
        GraphElement( t_fg, t_bg ), m_center( t_center ), radius( t_radius ) {}

    void draw()
    {
    	int startY = m_center.y - radius < 0 ? 0 : m_center.y - radius;
    	int startX = m_center.x - radius < 0 ? 0 : m_center.x - radius;

    	int stopY = m_center.y + radius >= LCD_HEIGHT ? LCD_HEIGHT - 1 : m_center.y + radius;
    	int stopX = m_center.x + radius >= LCD_WIDTH ? LCD_WIDTH - 1 : m_center.x + radius;

    	int distanceX, distanceY, distanceFromCenter;

    	for (int y = startY; y <= stopY; y++)
    	{
    		for (int x = startX; x <= stopX; x++)
    		{
    			distanceX = abs(m_center.x - x);
    			distanceY = abs(m_center.y - y);
    			distanceFromCenter = sqrt(distanceX*distanceX + distanceY*distanceY);

    			if(distanceFromCenter == radius)
    				drawPixel(x, y);
    		}
    	}
    } // IMPLEMENT!!!
};

class Character : public GraphElement 
{
public:
    // position of character
    Point2D m_pos;
    // character
    char m_character;

    Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg ) : 
      GraphElement( t_fg, t_bg ), m_pos( t_pos ), m_character( t_char ) {};

    void draw()
    {
    	for(int y = 0; y < 8; y++)
    	{
    		for(int x = 0; x < 8; x++)
    		{
    			if(font8x8[(int)m_character][y] & (1 << x))
    				drawPixel(m_pos.x + x, m_pos.y + y);
    		}
    	}
    }; // IMPLEMENT!!!
};

class Line : public GraphElement
{
private:
	void drawLineLow(int x0, int y0, int x1, int y1)
	{
		int dx = x1 - x0;
		int dy = y1 - y0;
		int yi = 1;

		if (dy < 0)
		{
		    yi = -1;
		    dy = -dy;
		}

		int D = (2 * dy) - dx;
		int y = y0;

		for (int x = x0; x <= x1; x++)
		{
		    drawPixel(x, y);

		    if (D > 0)
		    {
		        y = y + yi;
		        D = D + (2 * (dy - dx));
		    }
		    else
		        D = D + 2*dy;
		}
	}

	void drawLineHigh(int x0, int y0, int x1, int y1)
	{
		int dx = x1 - x0;
		int dy = y1 - y0;
		int xi = 1;

		if (dx < 0)
		{
		    xi = -1;
		    dx = -dx;
		}

		int D = (2 * dx) - dy;
		int x = x0;

		for (int y = y0; y <= y1; y++)
		{
		    drawPixel(x, y);

		    if (D > 0)
		    {
		        x = x + xi;
		        D = D + (2 * (dx - dy));
		    }
		    else
		        D = D + 2*dx;
		}
	}

public:
    // the first and the last point of line
    Point2D m_pos1, m_pos2;

    Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg ) : 
      GraphElement( t_fg, t_bg ), m_pos1( t_pos1 ), m_pos2( t_pos2 ) {}

    void draw()
    {
    	// Bresenham's line algorithm

    	if (abs(m_pos2.y - m_pos1.y) < abs(m_pos1.x - m_pos2.x))
		{
    	    if (m_pos1.x > m_pos2.x)
    	        drawLineLow(m_pos2.x, m_pos2.y, m_pos1.x, m_pos1.y);
    	    else
    	        drawLineLow(m_pos1.x, m_pos1.y, m_pos2.x, m_pos2.y);
		}
    	else
    	{
    	    if (m_pos1.y > m_pos2.y)
    	        drawLineHigh(m_pos2.x, m_pos2.y, m_pos1.x, m_pos1.y);
    	    else
    	        drawLineHigh(m_pos1.x, m_pos1.y, m_pos2.x, m_pos2.y);
		}
    }; // IMPLEMENT!!!
};

