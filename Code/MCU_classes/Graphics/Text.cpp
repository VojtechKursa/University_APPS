#include "Text.hpp"

Text::Text( Point2D t_pos, string t_characters, RGB t_fg, RGB t_bg ) : Text(t_pos, t_characters, t_fg, t_bg, false)
{}

Text::Text( Point2D t_pos, string t_characters, RGB t_fg, RGB t_bg, bool t_vertical ) :
    GraphElement( t_fg, t_bg ), m_pos( t_pos ), m_characters( t_characters ), vertical(t_vertical)
{}

void Text::draw()
{
    int i = 0;
    char character = m_characters[i];

    while(character != 0)
    {
        for(int y = 0; y < 20; y++)
        {
            for(int x = 0; x < 12; x++)
            {
                if(font[(int)character][y] & (65536 >> x))
                {
                    if (!vertical)
                        drawPixel(m_pos.x + (FONT_WIDTH * i) + x, m_pos.y + y);
                    else
                        drawPixel(m_pos.x + x, m_pos.y + (FONT_HEIGHT * i) + y);
                }
            }
        }

        character = m_characters[++i];
    }
};