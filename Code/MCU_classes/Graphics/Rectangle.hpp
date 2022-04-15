#ifndef _RECTANGLE_HPP
#define _RECTANGLE_HPP

#include "structs.h"
#include "GraphElement.hpp"
#include "Line.hpp"

class Rectangle : public GraphElement
{
public:
    Point2D ul_pos, lr_pos;

    Rectangle( Point2D ul_pos, Point2D lr_pos, RGB t_fg, RGB t_bg );

    void draw();
};

#endif