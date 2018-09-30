#pragma once

#include <QObject>

struct CompilationResult {
	bool compilationOk = true;
	int lineOfFirstError = 0;
	QByteArray compilation;
};

class CellComputerCompiler
	: public QObject
{
	Q_OBJECT
public:
	CellComputerCompiler(QObject* parent = nullptr) : QObject(parent) {}
	virtual ~CellComputerCompiler() = default;

	virtual CompilationResult compileSourceCode(std::string const& code) const = 0;
	virtual std::string decompileSourceCode(QByteArray const& data) const = 0;
};

