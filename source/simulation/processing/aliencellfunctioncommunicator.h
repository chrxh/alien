#ifndef ALIENCELLFUNCTIONCOMMUNICATOR_H
#define ALIENCELLFUNCTIONCOMMUNICATOR_H

#include "aliencellfunction.h"

class AlienCellFunctionCommunicator : public AlienCellFunction
{
public:
    AlienCellFunctionCommunicator ();
    AlienCellFunctionCommunicator (quint8* cellTypeData);
    AlienCellFunctionCommunicator (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName ();

};

#endif // ALIENCELLFUNCTIONCOMMUNICATOR_H
