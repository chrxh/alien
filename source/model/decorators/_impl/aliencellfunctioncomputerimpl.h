#ifndef ALIENTOKENFUNCTIONCOMPUTERIMPL_H
#define ALIENTOKENFUNCTIONCOMPUTERIMPL_H

#include <QByteArray>
#include <QChar>
#include <QVector>

#include "model/decorators/aliencellfunctioncomputer.h"

class AlienCellFunctionComputerImpl: public AlienCellFunctionComputer
{
public:
    AlienCellFunctionComputerImpl (AlienGrid*& grid);
    AlienCellFunctionComputerImpl (quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionComputerImpl (QDataStream& stream, AlienGrid*& grid);

    CellFunctionType getType () const { return CellFunctionType::COMPUTER; }
    void getInternalData (quint8* data);

    QString decompileInstructionCode () const;
    CompilationState injectAndCompileInstructionCode (QString code);
    QVector< quint8 >& getMemoryReference ();

protected:
    ProcessingResult processImpl (AlienToken* token, AlienCell* cell, AlienCell* previousCell);
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

#endif // ALIENTOKENFUNCTIONCOMPUTERIMPL_H
