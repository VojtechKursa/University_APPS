#include "Character.hpp"

Character::Character( Point2D t_pos, char t_char, RGB t_fg, RGB t_bg ) : 
    GraphElement( t_fg, t_bg ), m_pos( t_pos ), m_character( t_char )
    {}

void Character::draw()
{
    for(int y = 0; y < 20; y++)
    {
        for(int x = 0; x < 12; x++)
        {
            if(font[(int)m_character][y] & (65536 >> x))
                drawPixel(m_pos.x + x, m_pos.y + y);
        }
    }
}