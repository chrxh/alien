#include "aliencellfunctioncommunicator.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"

#include <QString>

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator()
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (quint8* cellTypeData)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (QDataStream& stream)
{

}

void AlienCellFunctionCommunicator::execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose)
{
    quint8 cmd = token->memory[static_cast<int>(COMMUNICATOR::IN)] % 4;
    quint8 channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
    quint8 msg = token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)];

    if( cmd == static_cast<int>(COMMUNICATOR_IN::SET_LISTENING_CHANNEL) )
        _listeningChannel = channel;
    if( cmd == static_cast<int>(COMMUNICATOR_IN::SEND_MESSAGE) ) {
        int numMsg = sendMessageToNearbyCellsAndReturnNumber(channel, msg, cell, grid);
        if( numMsg > 127)
            numMsg = 127;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_NUM_MESSAGE_SENT)] = numMsg;
    }
    if( cmd == static_cast<int>(COMMUNICATOR_IN::RECEIVE_MESSAGE) )
        receiveMessage();
}

QString AlienCellFunctionCommunicator::getCellFunctionName ()
{
    return "COMMUNICATOR";
}

void AlienCellFunctionCommunicator::setListeningChannelFromToken (AlienToken* token)
{
    _listeningChannel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
}

bool cellSelectCommunicatorFunction (AlienCell* cell)
{
    return cell->getCellFunction()->getCellFunctionName() == "COMMUNICATOR";
}

int AlienCellFunctionCommunicator::sendMessageToNearbyCellsAndReturnNumber (const quint8& channel, const quint8& msg, AlienCell* cell, AlienGrid* grid) const
{
    //find nearby communicator cells
    QVector3D cellPos = cell->calcPosition();
    qreal r = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE;
    QList< AlienCell* > nearbyCommunicatorCells = grid->getNearbySpecificCells(cellPos, r, cellSelectCommunicatorFunction);

    //send data to communicator cells
    int numMsg = 0;
    foreach(AlienCell* nearbyCell, nearbyCommunicatorCells) {
        if( nearbyCell != cell ) {
            if( sendMessageToCellAndReturnSuccess(channel, msg, nearbyCell) )
                ++numMsg;
        }
    }
    return numMsg;
}

bool AlienCellFunctionCommunicator::sendMessageToCellAndReturnSuccess (const quint8& channel, const quint8& msg, AlienCell* cell) const
{
    return true;
}

void AlienCellFunctionCommunicator::receiveMessage () const
{

}

