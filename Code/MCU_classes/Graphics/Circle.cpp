#include "Circle.hpp"



void Circle::drawPixel( int32_t t_x, int32_t t_y )
{
    if(t_x >= 0 && t_x < LCD_WIDTH && t_y >= 0 && t_y < LCD_HEIGHT)
        GraphElement::drawPixel(t_x, t_y);
}

void Circle::drawPoints(int xc, int yc, int x, int y)
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



Circle::Circle( Point2D t_center, int32_t t_radius, RGB t_fg, RGB t_bg ) : 
    GraphElement( t_fg, t_bg ), m_center( t_center ), radius( t_radius )
    {}



// Bresenham's Circle Algorithm
void Circle::draw()
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