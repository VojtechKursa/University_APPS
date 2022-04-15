#ifndef _RECTANGLE_WITH_DIAGONALS_HPP
#define _RECTANGLE_WITH_DIAGONALS_HPP

#include "Rectangle.hpp"
#include "Line.hpp"

class RectangleWithDiagonals : public Rectangle
{
public:
    RectangleWithDiagonals( Point2D ul_pos, Point2D lr_pos, RGB t_fg, RGB t_bg );

    void draw() override;
};

#endif