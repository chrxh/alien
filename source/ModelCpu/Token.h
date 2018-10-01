#pragma once

#include <QVector>

#include "Definitions.h"

#include "ModelInterface/Descriptions.h"

class Token
{
public:
	virtual ~Token() = default;

	virtual void setContext(UnitContext* context) = 0;

    virtual Token* duplicate () const = 0;

	virtual TokenDescription getDescription() const = 0;

	virtual int getTokenAccessNumber() const = 0;
	virtual void setTokenAccessNumber (int i) = 0;

	virtual void setEnergy (qreal energy) = 0;
	virtual qreal getEnergy () const = 0;

	virtual QByteArray& getMemoryRef () = 0;
};

