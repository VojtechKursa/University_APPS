#ifndef _CIRCLE_HPP
#define _CIRCLE_HPP

#include "GraphElement.hpp"
#include "structs.h"

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
    Point2D m_center; // Center of circle
    int32_t radius; // Radius of circle

    Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg );

    // Bresenham's Circle Algorithm
    void draw();
};

#endif