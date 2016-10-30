#ifndef ALIENCELLFUNCTION_H
#define ALIENCELLFUNCTION_H

#include "aliencelldecorator.h"

class AlienCellFunction: public AlienCellDecorator
{
public:
    AlienCellFunction (AlienCell* cell) : AlienCellDecorator(cell) {}
    virtual ~AlienCellFunction() {}

    virtual Type getType () const = 0;
    virtual void getInternalData (quint8* data) const = 0;

    enum class Type {
        COMPUTER,
        PROPULSION,
        SCANNER,
        WEAPON,
        CONSTRUCTOR,
        SENSOR,
        COMMUNICATOR,
        _COUNT
    };

protected:
    qreal calcAngle (AlienCell* origin, AlienCell* ref1, AlienCell* ref2) const;

    static qreal convertDataToAngle (quint8 b);
    static quint8 convertAngleToData (qreal a);
    static qreal convertDataToShiftLen (quint8 b);
    static quint8 convertShiftLenToData (qreal len);
    static quint8 convertURealToData (qreal r);
    static qreal convertDataToUReal (quint8 d);
    static quint8 convertIntToData (int i);

};

#endif // ALIENCELLFUNCTION_H

