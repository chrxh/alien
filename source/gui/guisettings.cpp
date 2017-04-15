#include "guisettings.h"

QFont GuiFunctions::getGlobalFont ()
{
    //set font
    QFont f(GLOBAL_FONT, 8, QFont::Bold);
    f.setStyleStrategy(QFont::PreferBitmap);
    return f;
}

QFont GuiFunctions::getCellFont()
{
	//set font
	QFont f(GLOBAL_FONT, 2, QFont::Normal);
	f.setStyleStrategy(QFont::PreferBitmap);
	return f;
}
