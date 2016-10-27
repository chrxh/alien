#include "guisettings.h"

QFont GuiFunctions::getGlobalFont ()
{
    //set font
    QFont f(GLOBAL_FONT, 9, QFont::Bold);
    f.setStyleStrategy(QFont::PreferBitmap);
    return f;
}
