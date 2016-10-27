#ifndef GLOBALFUNCTIONS_H
#define GLOBALFUNCTIONS_H

#include <QtGlobal>

class GlobalFunctions
{
public:
    static quint64 getTag ();
    static quint32 random (quint32 range);
    static qreal random (qreal min, qreal max);
};

#endif // GLOBALFUNCTIONS_H
