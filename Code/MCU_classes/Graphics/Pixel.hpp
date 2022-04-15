#ifndef _PIXEL_HPP
#define _PIXEL_HPP

#include "GraphElement.hpp"
#include "structs.h"

class Pixel : public GraphElement
{
public:
    // Position of Pixel
    Point2D m_pos;

    // constructor
    Pixel( Point2D t_pos, RGB t_fg_color, RGB t_bg_color ) : GraphElement( t_fg_color, t_bg_color ), m_pos( t_pos ) {}

    // Draw method implementation
    virtual void draw() { drawPixel( m_pos.x, m_pos.y ); }
};

#endif