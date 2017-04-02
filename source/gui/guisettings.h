#ifndef GUISETTINGS_H
#define GUISETTINGS_H

#include <QColor>
#include <QFont>

const QString BUTTON_STYLESHEET = "background-color: #202020; font-family: Courier New; font-weight: bold; font-size: 12px";
const QString TABLE_STYLESHEET = "background-color: #000000; color: #EEEEEE; gridline-color: #303030; selection-color: #EEEEEE; selection-background-color: #202020; font-family: Courier New; font-weight: bold; font-size: 12px;";
const QString SCROLLBAR_STYLESHEET = "background-color: #303030; color: #B0B0B0; gridline-color: #303030;";
const QColor BUTTON_TEXT_COLOR(0xC2, 0xC2, 0xC2);
const QColor BUTTON_TEXT_HIGHLIGHT_COLOR(0x90, 0x90, 0xFF);
const QString GLOBAL_FONT = "Courier New";
const qreal GRAPHICS_ITEM_SIZE = 10.0;
const QColor GRAPHICS_ITEM_COLOR(0x90, 0x90, 0x90);

//cell colors
const QColor INDIVIDUAL_CELL_COLOR1(0x50, 0x90, 0xFF, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR2(0xFF, 0x60, 0x40, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR3(0x70, 0xFF, 0x50, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR4(0xFF, 0xBF, 0x50, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR5(0xBF, 0x50, 0xFF, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR6(0x50, 0xFF, 0xEF, 0xB0);
const QColor INDIVIDUAL_CELL_COLOR7(0xBF, 0xBF, 0xBF, 0xB0);

class GuiFunctions
{
public:
    static QFont getGlobalFont ();
	static QFont getCellFont();
};

#endif // GUISETTINGS_H
