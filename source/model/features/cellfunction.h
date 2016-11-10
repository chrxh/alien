#ifndef CELLFUNCTION_H
#define CELLFUNCTION_H

#include "cellfeature.h"

#include "constants.h"

class CellFunction: public CellDecorator
{
public:
    CellFunction (Grid*& grid) : CellDecorator(grid) {}
    virtual ~CellFunction() {}

    void serialize (QDataStream& stream) const;

    //new interface
    virtual CellFunctionType getType () const = 0;
    virtual void getInternalData (quint8* data) const {}

protected:
    qreal calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const;

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

#endif // CELLFUNCTION_H

