#include "aliencellfunctioncommunicator.h"

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator()
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (quint8* cellTypeData)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (QDataStream& stream)
{

}

void AlienCellFunctionCommunicator::execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid*& space, AlienEnergy*& newParticle, bool& decompose)
{

}

QString AlienCellFunctionCommunicator::getCellFunctionName ()
{
    return "COMMUNICATOR";
}
