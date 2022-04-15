#ifndef _CHARACTER_HPP
#define _CHARACTER_HPP

#include "font12x20_msb.h"

#include "structs.h"
#include "GraphElement.hpp"
#include "graph_defines.h"

class Character : public GraphElement 
{
public:
    Point2D m_pos;	// position of character    
    char m_character;	// character

    Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg );

    void draw();
};

#endif