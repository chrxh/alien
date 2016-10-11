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
                                             AlienCell* cell,
                                             AlienCell* previousCell,
                                             AlienGrid* grid,
                                             AlienEnergy*& newParticle,
                                             bool& decompose)
{
    COMMUNICATOR_IN cmd = readCommandFromToken(token);
    if( cmd == COMMUNICATOR_IN::SET_LISTENING_CHANNEL )
        readListeningChannel(token);
    if( cmd == COMMUNICATOR_IN::SEND_MESSAGE )
        sendMessageToNearbyCommunicatorsAndUpdateToken(token, cell, previousCell, grid);
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

void AlienCellFunctionCommunicator::readListeningChannel (AlienToken* token)
{
    _listeningChannel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
}


void AlienCellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndUpdateToken (AlienToken* token,
                                                                            AlienCell* cell,
                                                                            AlienCell* previousCell,
                                                                            AlienGrid* grid) const
{
    quint8 channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
    quint8 msg = token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)];
    int numMsg = sendMessageToNearbyCommunicatorsAndReturnNumber(channel, msg, cell, previousCell, grid);
    token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)] = convertIntToData(numMsg);
}

void AlienCellFunctionCommunicator::receiveMessage () const
{
    if( _receivedNewMessage ) {
        _receivedNewMessage = false;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NEW_MESSAGE);
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_MESSAGE)]
                = _receivedMessage;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_ANGLE)]
                = _receivedAngle;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_DISTANCE)]
                = _receivedDistance;
    }
    else
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO_NEW_MESSAGE);
}

int AlienCellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndReturnNumber (const quint8& channel,
                                                                            const quint8& msg,
                                                                            AlienCell* cell,
                                                                            AlienCell* previousCell,
                                                                            AlienGrid* grid) const
{
    int numMsg = 0;
    QList< AlienCell* > nearbyCommunicatorCells = findNearbyCommunicator (cell, grid);
    foreach(AlienCell* nearbyCell, nearbyCommunicatorCells)
        if( nearbyCell != cell )
            if( sendMessageToCommunicatorAndReturnSuccess(channel, msg, cell, previousCell, nearbyCell, grid) )
                ++numMsg;
    return numMsg;
}

QList< AlienCell* > AlienCellFunctionCommunicator::findNearbyCommunicator(AlienCell* cell,
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

bool AlienCellFunctionCommunicator::sendMessageToCommunicatorAndReturnSuccess (const quint8& channel,
                                                                       const quint8& msg,
                                                                       AlienCell* senderCell,
                                                                       AlienCell* senderPreviousCell,
                                                                       AlienCell* receiverCell,
                                                                       AlienGrid* grid) const
{
    AlienCellFunctionCommunicator* communicator = getCommunicator(receiverCell);
    if( communicator ) {
        if( communicator->_listeningChannel == channel ) {
            communicator->_receivedMessage = msg;
            communicator->_receivedAngle = calcAngle(senderCell, senderPreviousCell, receiverCell, grid);
            communicator->_receivedDistance = convertURealToData(grid->distance(receiverCell, senderCell));
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

