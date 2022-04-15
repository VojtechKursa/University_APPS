#ifndef _GRAPHELEMENT_HPP
#define _GRAPHELEMENT_HPP

#include "lcd_lib.h"

class GraphElement
{
public:
    // foreground and background color
    RGB m_fg_color, m_bg_color;

    // constructor
    GraphElement( RGB t_fg_color, RGB t_bg_color ) : 
        m_fg_color( t_fg_color ), m_bg_color( t_bg_color ) {}

    // ONLY ONE INTERFACE WITH LCD HARDWARE!!!
    void drawPixel( int32_t t_x, int32_t t_y ) { lcd_put_pixel( t_x, t_y, convert_RGB888_to_RGB565( m_fg_color ) ); }
    
    // Draw graphics element
    virtual void draw() = 0;
    
    // Hide graphics element
    virtual void hide() { swap_fg_bg_color(); draw(); swap_fg_bg_color(); }


private:
    // swap foreground and backgroud colors
    void swap_fg_bg_color() { RGB l_tmp = m_fg_color; m_fg_color = m_bg_color; m_bg_color = l_tmp; } 

    // conversion of 24-bit RGB color into 16-bit color format
    uint16_t convert_RGB888_to_RGB565( RGB t_color )
    {
    	return (uint16_t)(((t_color.r & 0xF8) << 8) | ((t_color.g & 0xFC) << 3) | ((t_color.b & 0xF8) >> 3));
    }
};

#endif