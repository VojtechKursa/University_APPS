#include "RectangleWithDiagonals.hpp"

RectangleWithDiagonals::RectangleWithDiagonals( Point2D ul_pos, Point2D lr_pos, RGB t_fg, RGB t_bg ) :
    Rectangle(ul_pos, lr_pos, t_fg, t_bg)
{}

void RectangleWithDiagonals::draw() override
{
    Rectangle::draw();

    Line d1(ul_pos, lr_pos, m_fg_color, m_bg_color);
    Line d2({ul_pos.x, lr_pos.y}, {lr_pos.x, ul_pos.y}, m_fg_color, m_bg_color);

    d1.draw();
    d2.draw();
};