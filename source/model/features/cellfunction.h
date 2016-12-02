#ifndef CELLFUNCTION_H
#define CELLFUNCTION_H

#include "cellfeature.h"

#include "cellfeatureconstants.h"
#include "model/definitions.h"

class CellFunction: public CellFeature
{
public:
    CellFunction (Grid* grid) : CellFeature(grid) {}
    virtual ~CellFunction() {}

    //new interface
    virtual CellFunctionType getType () const = 0;
    virtual void getInternalData (quint8* ptr) const {}

protected:
    qreal calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const;

    static qreal convertDataToAngle (quint8 b);
    static quint8 convertAngleToData (qreal a);
    static qreal convertDataToShiftLen (quint8 b);
    static quint8 convertShiftLenToData (qreal len);
    static quint8 convertURealToData (qreal r);
    static qreal convertDataToUReal (quint8 d);
    static quint8 convertIntToData (int i);
};

#endif // CELLFUNCTION_H

