#ifndef ALIENTOKENFUNCTIONCOMPUTER_H
#define ALIENTOKENFUNCTIONCOMPUTER_H

#include <QByteArray>
#include <QChar>

#include "aliencellfunction.h"

class AlienCellFunctionComputer: public AlienCellFunction
{
public:
    AlienCellFunctionComputer (bool randomData);
    AlienCellFunctionComputer (quint8* cellTypeData);
    AlienCellFunctionComputer (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCode ();
    bool compileCode (QString code, int& errorLine);
    QString getCellFunctionName () const;

    void serialize (QDataStream& stream);

    //constants for cell function programming
    enum class COMPUTER_OPERATION {
        MOV, ADD, SUB, MUL, DIV, XOR, OR, AND, IFG, IFGE, IFE, IFNE, IFLE, IFL, ELSE, ENDIF
    };
    enum class COMPUTER_OPTYPE {
        MEM, MEMMEM, CMEM, CONST
    };

protected:
    void getInternalData (quint8* data);

private:
    QByteArray _code;
    int _numInstr;

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
                            qint8& op2);
    quint8 convertToAddress (qint8 addr, quint32 size);
    bool isNameChar (const QChar& c);
};

#endif // ALIENTOKENFUNCTIONCOMPUTER_H
