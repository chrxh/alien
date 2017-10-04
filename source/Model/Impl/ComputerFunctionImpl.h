#pragma once

#include <QByteArray>
#include <QChar>
#include <QVector>

#include "Model/Local/ComputerFunction.h"

class ComputerFunctionImpl
	: public ComputerFunction
{
public:
    ComputerFunctionImpl (UnitContext* context);
    ComputerFunctionImpl (QByteArray data, UnitContext* context);

	virtual QByteArray getInternalData () const override;

	virtual QString decompileInstructionCode () const override;
	virtual CompilationState injectAndCompileInstructionCode (QString sourceCode) override;

	virtual QByteArray& getMemoryReference() override;

	virtual void mutateImpl() override;

	virtual void serializePrimitives (QDataStream& stream) const override;
	virtual void deserializePrimitives (QDataStream& stream) override;

protected:
	virtual ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) override;

private:
	enum class State {
		LOOKING_FOR_INSTR_START,
		LOOKING_FOR_INSTR_END,
		LOOKING_FOR_OP1_START,
		LOOKING_FOR_OP1_END,
		LOOKING_FOR_SEPARATOR,
		LOOKING_FOR_COMPARATOR,
		LOOKING_FOR_OP2_START,
		LOOKING_FOR_OP2_END
	};
	struct InstructionUncoded {
		bool readingFinished = false;
		QString name;
		QString operand1;
		QString operand2;
		QString comp;
	};
	struct InstructionCoded {
		Enums::ComputerOperation::Type operation;
		Enums::ComputerOptype::Type opType1;
		Enums::ComputerOptype::Type opType2;
		quint8 operand1;
		quint8 operand2;
	};
	bool resolveInstructionAndReturnSuccess(InstructionCoded& instructionCoded, InstructionUncoded instructionUncoded);
	bool gotoNextStateAndReturnSuccess(State &state, QChar &currentSymbol, InstructionUncoded& instruction, int symbolPos, int codeSize);

	void writeInstruction(InstructionCoded const& instructionCoded);
	void readInstruction(int& instructionPointer, InstructionCoded& instructionCoded) const;

	QString applyTableToCode(QString s);

    QByteArray _code;
	QByteArray _memory;
};

