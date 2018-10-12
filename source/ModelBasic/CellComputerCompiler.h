#pragma once

#include <QObject>

#include "Definitions.h"

struct CompilationResult {
	bool compilationOk = true;
	int lineOfFirstError = 0;
	QByteArray compilation;
};

class MODELBASIC_EXPORT CellComputerCompiler
	: public QObject
{
	Q_OBJECT
public:
	CellComputerCompiler(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CellComputerCompiler() = default;

	virtual CompilationResult compileSourceCode(std::string const& code) const = 0;
	virtual std::string decompileSourceCode(QByteArray const& data) const = 0;
};

