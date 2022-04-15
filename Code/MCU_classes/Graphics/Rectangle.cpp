#include "Rectangle.hpp"

Rectangle::Rectangle( Point2D ul_pos, Point2D lr_pos, RGB t_fg, RGB t_bg ) :
    GraphElement( t_fg, t_bg ), ul_pos( ul_pos ), lr_pos( lr_pos )
{}

void Rectangle::draw()
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