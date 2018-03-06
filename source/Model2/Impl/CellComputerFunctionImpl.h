#pragma once

#include <QByteArray>
#include <QChar>
#include <QVector>

#include "Model/Api/CellComputerCompiler.h"
#include "Model/Local/CellComputerFunction.h"

class CellComputerFunctionImpl
	: public CellComputerFunction
{
public:
    CellComputerFunctionImpl (UnitContext* context);
    CellComputerFunctionImpl (QByteArray const& data, UnitContext* context);

	virtual QByteArray getInternalData () const override;

	static CompilationResult compileSourceCode(std::string const& code, SymbolTable const* symbols);
	static std::string decompileSourceCode(QByteArray const& data, SimulationParameters const* parameters);

	virtual void mutateImpl() override;

protected:
	virtual ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) override;
	virtual void appendDescriptionImpl(CellFeatureDescription & desc) const override;

private:
	QByteArray _code;
	QByteArray _memory;
};

