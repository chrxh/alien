#include "Settings.h"

QFont GuiSettings::getGlobalFont ()
{
    //set font
    QFont f(GLOBAL_FONT, 8, QFont::Bold);
    f.setStyleStrategy(QFont::PreferBitmap);
    return f;
}

QFont GuiSettings::getCellFont()
{
	//set font
	QFont f(GLOBAL_FONT, 2, QFont::Normal);
	f.setStyleStrategy(QFont::PreferBitmap);
	return f;
}
