#ifndef CELLFUNCTION_H
#define CELLFUNCTION_H

#include "cellfeature.h"

#include "cellfeatureconstants.h"
#include "model/definitions.h"

class CellFunction: public CellFeature
{
public:
    CellFunction (SimulationContext* context) : CellFeature(context) {}
    virtual ~CellFunction() {}

    //new interface
    virtual CellFunctionType getType () const = 0;
    virtual void getInternalData (quint8* ptr) const {}

protected:
    qreal calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const;
};

#endif // CELLFUNCTION_H

