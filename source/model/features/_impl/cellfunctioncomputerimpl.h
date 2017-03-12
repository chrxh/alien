#ifndef TOKENFUNCTIONCOMPUTERIMPL_H
#define TOKENFUNCTIONCOMPUTERIMPL_H

#include <QByteArray>
#include <QChar>
#include <QVector>

#include "model/features/cellfunctioncomputer.h"

class CellFunctionComputerImpl: public CellFunctionComputer
{
public:
    CellFunctionComputerImpl (SimulationContext* context);
    CellFunctionComputerImpl (QByteArray data, SimulationContext* context);

	QByteArray getInternalData () const override;

    QString decompileInstructionCode () const override;
    CompilationState injectAndCompileInstructionCode (QString sourceCode) override;

	QByteArray& getMemoryReference() override;

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives (QDataStream& stream) override;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

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
		COMPUTER_OPERATION operation;
		COMPUTER_OPTYPE opType1;
		COMPUTER_OPTYPE opType2;
		quint8 operand1;
		quint8 operand2;
	};
	bool resolveInstruction(InstructionCoded& instructionCoded, InstructionUncoded instructionUncoded);
	bool stateMachine(State &state, QChar &currentSymbol, InstructionUncoded& instruction, int symbolPos, int codeSize);

	void writeInstruction(InstructionCoded const& instructionCoded);
	void readInstruction(int& instructionPointer, InstructionCoded& instructionCoded) const;

	QString applyTableToCode(QString s);

    QByteArray _code;
	QByteArray _memory;
	SymbolTable* _symbolTable;
};

#endif // TOKENFUNCTIONCOMPUTERIMPL_H
