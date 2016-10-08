#include "aliencellfunctioncommunicator.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"

#include <QString>

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator()
    : _listeningChannel(0),
      _receivedNewMessage(false),
      _receivedMessage(0),
      _receivedAngle(0),
      _receivedDistance(0)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (quint8* cellTypeData)
    : AlienCellFunctionCommunicator()
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (QDataStream& stream)
    : AlienCellFunctionCommunicator()
{

}

void AlienCellFunctionCommunicator::execute (AlienToken* token,
                                             AlienCell* previousCell,
                                             AlienCell* cell,
                                             AlienGrid* grid,
                                             AlienEnergy*& newParticle,
                                             bool& decompose)
{
    COMMUNICATOR_IN cmd = readCommandFromToken(token);
    if( cmd == COMMUNICATOR_IN::SET_LISTENING_CHANNEL )
        _listeningChannel = readListeningChannelFrom(token);
    if( cmd == COMMUNICATOR_IN::SEND_MESSAGE )
        sendMessageToNearbyCellsAndUpdateToken(token, cell, grid);
    if( cmd == COMMUNICATOR_IN::RECEIVE_MESSAGE )
        receiveMessage();
}

QString AlienCellFunctionCommunicator::getCellFunctionName () const
{
    return "COMMUNICATOR";
}

AlienCellFunctionCommunicator::COMMUNICATOR_IN AlienCellFunctionCommunicator::readCommandFromToken (AlienToken* token) const
{
    return static_cast<COMMUNICATOR_IN>(token->memory[static_cast<int>(COMMUNICATOR::IN)] % 4);
}

void AlienCellFunctionCommunicator::sendMessageToNearbyCellsAndUpdateToken (AlienToken* token,
                                                                            AlienCell* cell,
                                                                            AlienGrid* grid) const
{
    quint8 channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
    quint8 msg = token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)];
    int numMsg = sendMessageToNearbyCellsAndReturnNumber(channel, msg, cell, grid);
    token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)] = convertIntToData(numMsg);
}

quint8 AlienCellFunctionCommunicator::readListeningChannelFrom (AlienToken* token) const
{
    return token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
}


int AlienCellFunctionCommunicator::sendMessageToNearbyCellsAndReturnNumber (const quint8& channel,
                                                                            const quint8& msg,
                                                                            AlienCell* cell,
                                                                            AlienGrid* grid) const
{
    int numMsg = 0;
    QList< AlienCell* > nearbyCommunicatorCells = findNearbyCommunicatorCells(cell, grid);
    foreach(AlienCell* nearbyCell, nearbyCommunicatorCells)
        if( nearbyCell != cell )
            if( sendMessageToCellAndReturnSuccess(channel, msg, cell, nearbyCell, grid) )
                ++numMsg;
    return numMsg;
}

QList< AlienCell* > AlienCellFunctionCommunicator::findNearbyCommunicatorCells (AlienCell* cell,
                                                                                AlienGrid* grid) const
{
    AlienGrid::CellSelectFunction cellSelectCommunicatorFunction =
        [](AlienCell* cell)
        {
            return cell->getCellFunction()->getCellFunctionName() == "COMMUNICATOR";
        };
    QVector3D cellPos = cell->calcPosition();
    qreal range = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE;
    return grid->getNearbySpecificCells(cellPos, range, cellSelectCommunicatorFunction);
}

bool AlienCellFunctionCommunicator::sendMessageToCellAndReturnSuccess (const quint8& channel,
                                                                       const quint8& msg,
                                                                       AlienCell* senderCell,
                                                                       AlienCell* receiverCell,
                                                                       AlienGrid* grid) const
{
    AlienCellFunctionCommunicator* communicator = getCommunicator(receiverCell);
    if( communicator ) {
        if( communicator->_listeningChannel == channel ) {
            communicator->_receivedMessage = msg;
            communicator->_receivedAngle = 0; //mock
            communicator->_receivedDistance = convertURealToData(grid->calcDistance(senderCell, receiverCell));
            return true;
        }
    }
    return false;
}

AlienCellFunctionCommunicator* AlienCellFunctionCommunicator::getCommunicator (AlienCell* cell) const
{
    AlienCellFunction* cellFunction = cell->getCellFunction();
    if( cellFunction->getCellFunctionName() == "COMMUNICATOR" )
        return static_cast< AlienCellFunctionCommunicator* >(cellFunction);
    return 0;
}

void AlienCellFunctionCommunicator::receiveMessage () const
{

}

