#ifndef ALIENCELLFUNCTION_H
#define ALIENCELLFUNCTION_H

#include "aliencelldecorator.h"

#include "constants.h"

class AlienCellFunction: public AlienCellDecorator
{
public:
    AlienCellFunction (AlienGrid*& grid) : AlienCellDecorator(grid) {}
    virtual ~AlienCellFunction() {}

    void serialize (QDataStream& stream) const;

    //new interface
    virtual CellFunctionType getType () const = 0;
    virtual void getInternalData (quint8* data) const {}

protected:
    qreal calcAngle (AlienCell* origin, AlienCell* ref1, AlienCell* ref2) const;

public:
    static CellFunctionType getType (QDataStream& stream);
protected:
    static qreal convertDataToAngle (quint8 b);
    static quint8 convertAngleToData (qreal a);
    static qreal convertDataToShiftLen (quint8 b);
    static quint8 convertShiftLenToData (qreal len);
    static quint8 convertURealToData (qreal r);
    static qreal convertDataToUReal (quint8 d);
    static quint8 convertIntToData (int i);
};

#endif // ALIENCELLFUNCTION_H

