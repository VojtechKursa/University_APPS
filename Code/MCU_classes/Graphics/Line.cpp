#include "Line.hpp"

void Line::drawLineLow(int x0, int y0, int x1, int y1)
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

void Line::drawLineHigh(int x0, int y0, int x1, int y1)
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

Line::Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg ) : 
    GraphElement( t_fg, t_bg ), m_pos1( t_pos1 ), m_pos2( t_pos2 )
{}

// Bresenham's line algorithm
void Line::draw()
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
};