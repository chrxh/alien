#pragma once

#include "Model/Api/Definitions.h"
#include "Model/Api/CellFeatureEnums.h"

#include "CellFeature.h"

class CellFunction
	: public CellFeature
{
public:
    CellFunction (UnitContext* context) : CellFeature(context) {}
    virtual ~CellFunction() {}

    //new interface
    virtual Enums::CellFunction::Type getType () const = 0;
	virtual QByteArray getInternalData() const {
		return QByteArray();
	}


protected:
    qreal calcAngle (Cell* origin, Cell* ref1, Cell* ref2) const;
};

