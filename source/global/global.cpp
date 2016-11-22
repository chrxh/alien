#include "global.h"
#include <QMutex>

QMutex _mutex;

namespace {
    quint64 _tag(0);
}

quint64 GlobalFunctions::createNewTag ()
{
    _mutex.lock();
    quint64 tag = ++_tag;
    _mutex.unlock();
    return tag;
}

quint64 GlobalFunctions::getTag ()
{
    _mutex.lock();
    quint64 tag = _tag;
    _mutex.unlock();
    return tag;
}

void GlobalFunctions::setTag (quint64 tag)
{
    _mutex.lock();
    _tag = tag;
    _mutex.unlock();
}

quint32 GlobalFunctions::random (quint32 range)
{
    return (quint32)((qreal)range*(qreal)qrand()/RAND_MAX);
}

qreal GlobalFunctions::random (qreal min, qreal max)
{
    return (qreal)GlobalFunctions::random((max-min)*1000)/1000.0+min;
}

