#ifndef ALIENCELLFUNCTION_H
#define ALIENCELLFUNCTION_H

#include "aliencelldecorator.h"

#include "constants.h"

class AlienCellFunction: public AlienCellDecorator
{
public:
    AlienCellFunction (AlienCell* cell, AlienGrid*& grid) : AlienCellDecorator(cell, grid) {}
    virtual ~AlienCellFunction() {}

    virtual CellFunctionType getType () const = 0;
    virtual void getInternalData (quint8* data) const {}

    static CellFunctionType getType (QDataStream& stream);
    void serialize (QDataStream& stream) const;

protected:
    virtual void serializeInternalData (QDataStream& stream) const {}

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

