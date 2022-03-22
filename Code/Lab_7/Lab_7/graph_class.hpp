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

#include "font12x20_msb.h"
#include <math.h>

#define font font
#define FONT_WIDTH 12
#define FONT_HEIGHT 20

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
private:
	void drawPixel( int32_t t_x, int32_t t_y )
	{
		if(t_x >= 0 && t_x < LCD_WIDTH && t_y >= 0 && t_y < LCD_HEIGHT)
			GraphElement::drawPixel(t_x, t_y);
	}

	void drawPoints(int xc, int yc, int x, int y)
    {
		drawPixel(xc+x, yc+y);
		drawPixel(xc-x, yc+y);
		drawPixel(xc+x, yc-y);
		drawPixel(xc-x, yc-y);
		drawPixel(xc+y, yc+x);
		drawPixel(xc-y, yc+x);
		drawPixel(xc+y, yc-x);
		drawPixel(xc-y, yc-x);
    }

public:
    // Center of circle
    Point2D m_center;
    // Radius of circle
    int32_t radius;

    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) : 
        GraphElement( t_fg, t_bg ), m_center( t_center ), radius( t_radius ) {}

    // Bresenham's Circle Algorithm
    void draw()
    {
        int x = 0, y = radius;
        int d = 3 - 2 * radius;

        drawPoints(m_center.x, m_center.y, x, y);

        while (y >= x)
        {
            x++;

            if (d > 0)
            {
                y--;
                d = d + 4 * (x - y) + 10;
            }
            else
                d = d + 4 * x + 6;

            drawPoints(m_center.x, m_center.y, x, y);
        }
    }
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
    	for(int y = 0; y < 20; y++)
    	{
    		for(int x = 0; x < 12; x++)
    		{
    			if(font[(int)m_character][y] & (65536 >> x))
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

    // Bresenham's line algorithm
    void draw()
    {
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

class Text : public GraphElement
{
public:
    Point2D m_pos;
    string m_characters;
    bool vertical;

    Text( Point2D t_pos, string t_characters, RGB t_fg, RGB t_bg ) : Text(t_pos, t_characters, t_fg, t_bg, false)
    	{};
    Text( Point2D t_pos, string t_characters, RGB t_fg, RGB t_bg, bool t_vertical ) :
    	GraphElement( t_fg, t_bg ), m_pos( t_pos ), m_characters( t_characters ), vertical(t_vertical) {};

    void draw()
    {
    	int i = 0;
    	char character = m_characters[i];

    	while(character != 0)
    	{
			for(int y = 0; y < 20; y++)
			{
				for(int x = 0; x < 12; x++)
				{
					if(font[(int)character][y] & (65536 >> x))
					{
						if (!vertical)
							drawPixel(m_pos.x + (FONT_WIDTH * i) + x, m_pos.y + y);
						else
							drawPixel(m_pos.x + x, m_pos.y + (FONT_HEIGHT * i) + y);
					}
				}
			}

			character = m_characters[++i];
    	}
    };
};

class Rectangle : public GraphElement
{
public:
    Point2D ul_pos, lr_pos;

    Rectangle( Point2D ul_pos, Point2D lr_pos, RGB t_fg, RGB t_bg ) :
    	GraphElement( t_fg, t_bg ), ul_pos( ul_pos ), lr_pos( lr_pos ) {};

    void draw()
    {
    	Line upperLine(ul_pos, {lr_pos.x, ul_pos.y}, m_fg_color, m_bg_color);
    	Line leftLine(ul_pos, {ul_pos.x, lr_pos.y}, m_fg_color, m_bg_color);
    	Line lowerLine(lr_pos, {ul_pos.x, lr_pos.y}, m_fg_color, m_bg_color);
    	Line rightLine(lr_pos, {lr_pos.x, ul_pos.y}, m_fg_color, m_bg_color);

    	upperLine.draw();
    	leftLine.draw();
    	lowerLine.draw();
    	rightLine.draw();
    };
};

class RectangleWithDiagonals : public Rectangle
{
public:
    RectangleWithDiagonals( Point2D ul_pos, Point2D lr_pos, RGB t_fg, RGB t_bg ) :
    	Rectangle(ul_pos, lr_pos, t_fg, t_bg) {};

    void draw() override
    {
    	Rectangle::draw();

    	Line d1(ul_pos, lr_pos, m_fg_color, m_bg_color);
    	Line d2({ul_pos.x, lr_pos.y}, {lr_pos.x, ul_pos.y}, m_fg_color, m_bg_color);

    	d1.draw();
    	d2.draw();
    };
};

