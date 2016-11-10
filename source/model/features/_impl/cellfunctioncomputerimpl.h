#ifndef TOKENFUNCTIONCOMPUTERIMPL_H
#define TOKENFUNCTIONCOMPUTERIMPL_H

#include <QByteArray>
#include <QChar>
#include <QVector>

#include "model/features/cellfunctioncomputer.h"

class CellFunctionComputerImpl: public CellFunctionComputer
{
public:
    CellFunctionComputerImpl (Grid*& grid);
    CellFunctionComputerImpl (quint8* cellFunctionData, Grid*& grid);
    CellFunctionComputerImpl (QDataStream& stream, Grid*& grid);

    CellFunctionType getType () const { return CellFunctionType::COMPUTER; }
    void getInternalData (quint8* data);

    QString decompileInstructionCode () const;
    CompilationState injectAndCompileInstructionCode (QString code);
    QVector< quint8 >& getMemoryReference ();

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
    void serializeImpl (QDataStream& stream) const;

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
    int _numInstr;
    QVector< quint8 > _memory;
};

#endif // TOKENFUNCTIONCOMPUTERIMPL_H
