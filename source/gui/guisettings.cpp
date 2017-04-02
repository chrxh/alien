#include "guisettings.h"

QFont GuiFunctions::getGlobalFont ()
{
    //set font
    QFont f(GLOBAL_FONT, 9, QFont::Bold);
    f.setStyleStrategy(QFont::PreferBitmap);
    return f;
}

QFont GuiFunctions::getCellFont()
{
	//set font
	QFont f(GLOBAL_FONT, 3, QFont::Normal);
	f.setStyleStrategy(QFont::PreferBitmap);
	return f;
}
