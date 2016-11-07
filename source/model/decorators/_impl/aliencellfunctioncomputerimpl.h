#ifndef ALIENTOKENFUNCTIONCOMPUTERIMPL_H
#define ALIENTOKENFUNCTIONCOMPUTERIMPL_H

#include <QByteArray>
#include <QChar>

#include "model/decorators/aliencellfunctioncomputer.h"

class AlienCellFunctionComputerImpl: public AlienCellFunctionComputer
{
public:
    AlienCellFunctionComputerImpl (AlienCell* cell, bool randomData, AlienGrid*& grid);
    AlienCellFunctionComputerImpl (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionComputerImpl (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::COMPUTER; }
    QString decompileInstructionCode () const;
    virtual CompilationState injectAndCompileInstructionCode (QString code);

    void serialize (QDataStream& stream);
    void getInternalData (quint8* data);

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
};

#endif // ALIENTOKENFUNCTIONCOMPUTERIMPL_H
