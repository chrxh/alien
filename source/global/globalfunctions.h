#ifndef GLOBALFUNCTIONS_H
#define GLOBALFUNCTIONS_H

#include <QtGlobal>
#include <QFont>

class GlobalFunctions
{
public:
    static quint64 getTag ();
    static quint32 random (quint32 range);
    static qreal random (qreal min, qreal max);
    static QFont getGlobalFont ();
};

#endif // GLOBALFUNCTIONS_H
