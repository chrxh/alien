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
	QByteArray& getMemoryReference () override;

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives (QDataStream& stream) override;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:

    void codeInstruction (int& instructionPointer, quint8 instr, quint8 opTyp1, quint8 opTyp2
		, qint8 op1, qint8 op2);
    void decodeInstruction (int& instructionPointer, quint8& instr, quint8& opTyp1, quint8& opTyp2
		, qint8& op1, qint8& op2) const;

    QByteArray _code;
	QByteArray _memory;
	SymbolTable* _symbolTable;
};

#endif // TOKENFUNCTIONCOMPUTERIMPL_H
