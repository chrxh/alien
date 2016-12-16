#ifndef GLOBALFUNCTIONS_H
#define GLOBALFUNCTIONS_H

#include <QtGlobal>

class GlobalFunctions
{
public:
    static quint64 createNewTag ();
    static quint64 getTag ();
    static void setTag (quint64 tag);
    static quint32 random (quint32 range);
    static qreal random (qreal min, qreal max);
	static qreal random ();
};

#endif // GLOBALFUNCTIONS_H
