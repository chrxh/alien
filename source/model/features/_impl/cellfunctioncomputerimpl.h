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

    void getInternalData (quint8* data) const;

    QString decompileInstructionCode () const;
    CompilationState injectAndCompileInstructionCode (QString code);
    QVector< quint8 >& getMemoryReference ();

    void serialize (QDataStream& stream) const;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);

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
