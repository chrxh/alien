#pragma once

#include <QByteArray>
#include <QChar>
#include <QVector>

#include "ModelBasic/CellComputerCompiler.h"
#include "CellComputerFunction.h"

class CellComputerFunctionImpl
	: public CellComputerFunction
{
public:
    CellComputerFunctionImpl (QByteArray const& code, QByteArray const& memory, UnitContext* context);

	virtual QByteArray getInternalData () const override;

	virtual void mutateImpl() override;

protected:
	virtual ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) override;
	virtual void appendDescriptionImpl(CellFeatureDescription & desc) const override;

private:
	QByteArray _code;
	QByteArray _memory;
};

