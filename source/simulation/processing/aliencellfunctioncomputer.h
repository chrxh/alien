#ifndef ALIENTOKENFUNCTIONCOMPUTER_H
#define ALIENTOKENFUNCTIONCOMPUTER_H

#include <QByteArray>

#include "aliencellfunction.h"

class AlienCellFunctionComputer: public AlienCellFunction
{
public:
    AlienCellFunctionComputer (bool randomData);
    AlienCellFunctionComputer (quint8* cellTypeData);
    AlienCellFunctionComputer (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose);
    virtual QString getCode ();
    virtual bool compileCode (QString code, int& errorLine);
    QString getCellFunctionName ();

    void serialize (QDataStream& stream);

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
