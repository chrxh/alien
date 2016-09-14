#include "globalfunctions.h"
#include <QMutex>

#include "guisettings.h"

QMutex _mutex;

quint64 _tag(0);

quint64 GlobalFunctions::getTag ()
{
    _mutex.lock();
    quint64 tag(++_tag);
    _mutex.unlock();
    return tag;
}

quint32 GlobalFunctions::random (quint32 range)
{
    return (quint32)((qreal)range*(qreal)qrand()/RAND_MAX);
}

qreal GlobalFunctions::random (qreal min, qreal max)
{
    return (qreal)GlobalFunctions::random((max-min)*1000)/1000.0+min;
}

QFont GlobalFunctions::getGlobalFont ()
{
    //set font
    QFont f(GLOBAL_FONT, 9, QFont::Bold);
    f.setStyleStrategy(QFont::PreferBitmap);
    return f;
}

