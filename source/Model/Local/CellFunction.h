#pragma once

#include "Model/Api/Definitions.h"
#include "Model/Api/CellFeatureEnums.h"

#include "CellFeatureChain.h"

class CellFunction
	: public CellFeatureChain
{
public:
    CellFunction (UnitContext* context) : CellFeatureChain(context) {}
    virtual ~CellFunction() {}

    //new interface
    virtual Enums::CellFunction::Type getType () const = 0;
	virtual QByteArray getInternalData() const;

protected:
	virtual void appendDescriptionImpl(CellFeatureDescription & desc) const override;
	qreal calcAngle(Cell* origin, Cell* ref1, Cell* ref2) const;
};

