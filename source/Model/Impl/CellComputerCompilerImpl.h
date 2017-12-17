#pragma once

#include "Model/Api/Definitions.h"
#include "Model/Local/CellComputerCompilerLocal.h"

class CellComputerCompilerImpl
	: public CellComputerCompilerLocal
{
	Q_OBJECT
public:
	CellComputerCompilerImpl(QObject * parent = nullptr);
	virtual ~CellComputerCompilerImpl() = default;

	virtual void init(SymbolTable const* symbols, SimulationParameters const* parameters) override;

	virtual CompilationResult compileSourceCode(std::string const& code) const override;
	virtual std::string decompileSourceCode(QByteArray const& data) const override;

private:
	SymbolTable const* _symbols = nullptr;
	SimulationParameters const* _parameters = nullptr;
};
