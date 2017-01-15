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
    CellFunctionComputerImpl (quint8* cellFunctionData, SimulationContext* context);

    void getInternalData (quint8* data) const override;

    QString decompileInstructionCode () const override;
    CompilationState injectAndCompileInstructionCode (QString code) override;
    QVector< quint8 >& getMemoryReference () override;

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives (QDataStream& stream) override;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:

    void codeInstruction (int& instructionPointer,
                          quint8 instr,
                          quint8 opTyp1,
                          quint8 opTyp2,
                          qint8 op1,
                          qint8 op2);
    void decodeInstruction (int& instructionPointer,
                            quint8& instr,
                            quint8& opTyp1,
                            quint8& opTyp2,
                            qint8& op1,
                            qint8& op2) const;

    QByteArray _code;
    int _numInstr = 0;
    QVector< quint8 > _memory;
	SymbolTable* _symbolTable;
};

#endif // TOKENFUNCTIONCOMPUTERIMPL_H
