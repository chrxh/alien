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
    CompilationState injectAndCompileInstructionCode (QString code) override;

	QByteArray& getMemoryReference() override;

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives (QDataStream& stream) override;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:
    void codeInstruction (quint8 instr, quint8 opTyp1, quint8 opTyp2, qint8 op1, qint8 op2);
    void decodeInstruction (int& instructionPointer, quint8& instr, quint8& opTyp1, quint8& opTyp2
		, qint8& op1, qint8& op2) const;

	enum State {
		LOOKING_FOR_INSTR_START,
		LOOKING_FOR_INSTR_END,
		LOOKING_FOR_OP1_START,
		LOOKING_FOR_OP1_END,
		LOOKING_FOR_SEPARATOR,
		LOOKING_FOR_COMPARATOR,
		LOOKING_FOR_OP2_START,
		LOOKING_FOR_OP2_END
	};
	struct Instruction {
		QString name;
		QString op1;
		QString op2;
		QString comp;
	};
	bool stateMachine(State &state, QChar &currentSymbol, Instruction& instruction, bool& instructionRead
		, int symbolPos, int codeSize);

    QByteArray _code;
	QByteArray _memory;
	SymbolTable* _symbolTable;
};

#endif // TOKENFUNCTIONCOMPUTERIMPL_H
