#ifndef _TEXT_HPP
#define _TEXT_HPP

#include "structs.h"
#include "GraphElement.hpp"

class Text : public GraphElement
{
public:
    Point2D m_pos;
    string m_characters;
    bool vertical;

    Text( Point2D t_pos, string t_characters, RGB t_fg, RGB t_bg );
    Text( Point2D t_pos, string t_characters, RGB t_fg, RGB t_bg, bool t_vertical );

    void draw();
};

#endif