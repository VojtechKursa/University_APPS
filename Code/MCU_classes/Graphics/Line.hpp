#ifndef _LINE_HPP
#define _LINE_HPP

#include <math.h>

#include "GraphElement.hpp"
#include "structs.h"

class Line : public GraphElement
{
private:
	void drawLineLow(int x0, int y0, int x1, int y1);
	void drawLineHigh(int x0, int y0, int x1, int y1);

public:
    // the first and the last point of line
    Point2D m_pos1, m_pos2;

    Line( Point2D t_pos1, Point2D t_pos2, RGB t_fg, RGB t_bg );

    // Bresenham's line algorithm
    void draw();
};

#endif