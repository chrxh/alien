#include "aliencellfunctioncommunicator.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"

#include <QString>

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator(AlienGrid*& grid)
    : AlienCellFunction(grid), _newMessageReceived(false)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (quint8* cellTypeData, AlienGrid*& grid)
    : AlienCellFunctionCommunicator(grid)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (QDataStream& stream, AlienGrid*& grid)
    : AlienCellFunctionCommunicator(grid)
{
    stream >> _newMessageReceived >> _receivedMessage.channel >> _receivedMessage.message >> _receivedMessage.angle >> _receivedMessage.distance;
}

void AlienCellFunctionCommunicator::execute (AlienToken* token,
                                             AlienCell* cell,
                                             AlienCell* previousCell,
                                             AlienEnergy*& newParticle,
                                             bool& decompose)
{
    COMMUNICATOR_IN cmd = readCommandFromToken(token);
    if( cmd == COMMUNICATOR_IN::SET_LISTENING_CHANNEL )
        setListeningChannel(token);
    if( cmd == COMMUNICATOR_IN::SEND_MESSAGE )
        sendMessageToNearbyCommunicatorsAndUpdateToken(token, cell, previousCell);
    if( cmd == COMMUNICATOR_IN::RECEIVE_MESSAGE )
        receiveMessage(token);
}

QString AlienCellFunctionCommunicator::getCellFunctionName () const
{
    return "COMMUNICATOR";
}

void AlienCellFunctionCommunicator::serialize (QDataStream& stream)
{
    AlienCellFunction::serialize(stream);
    stream << _newMessageReceived << _receivedMessage.channel << _receivedMessage.message << _receivedMessage.angle << _receivedMessage.distance;
}


AlienCellFunctionCommunicator::COMMUNICATOR_IN AlienCellFunctionCommunicator::readCommandFromToken (AlienToken* token) const
{
    return static_cast<COMMUNICATOR_IN>(token->memory[static_cast<int>(COMMUNICATOR::IN)] % 4);
}

void AlienCellFunctionCommunicator::setListeningChannel (AlienToken* token)
{
    _receivedMessage.channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
}


void AlienCellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndUpdateToken (AlienToken* token,
                                                                            AlienCell* cell,
                                                                            AlienCell* previousCell) const
{
    MessageData messageToSend;
    messageToSend.channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
    messageToSend.message = token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)];
    messageToSend.angle = token->memory[static_cast<int>(COMMUNICATOR::IN_ANGLE)];
    messageToSend.distance = token->memory[static_cast<int>(COMMUNICATOR::IN_DISTANCE)];
    int numMsg = sendMessageToNearbyCommunicatorsAndReturnNumber(messageToSend, cell, previousCell);
    token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)] = convertIntToData(numMsg);
}

void AlienCellFunctionCommunicator::receiveMessage (AlienToken* token)
{
    if( _newMessageReceived ) {
        _newMessageReceived = false;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NEW_MESSAGE);
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_MESSAGE)]
                = _receivedMessage.message;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_ANGLE)]
                = _receivedMessage.angle;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_DISTANCE)]
                = _receivedMessage.distance;
    }
    else
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO_NEW_MESSAGE);
}

int AlienCellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageToSend,
                                                                            AlienCell* senderCell,
                                                                            AlienCell* senderPreviousCell) const
{
    int numMsg = 0;
    QList< AlienCell* > nearbyCommunicatorCells = findNearbyCommunicator (senderCell);
    foreach(AlienCell* nearbyCell, nearbyCommunicatorCells)
        if( nearbyCell != senderCell )
            if( sendMessageToCommunicatorAndReturnSuccess(messageToSend, senderCell, senderPreviousCell, nearbyCell) )
                ++numMsg;
    return numMsg;
}

QList< AlienCell* > AlienCellFunctionCommunicator::findNearbyCommunicator(AlienCell* cell) const
{
    AlienGrid::CellSelectFunction cellSelectCommunicatorFunction =
        [](AlienCell* cell)
        {
            return cell->getCellFunction()->getCellFunctionName() == "COMMUNICATOR";
        };
    QVector3D cellPos = cell->calcPosition();
    qreal range = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE;
    return _grid->getNearbySpecificCells(cellPos, range, cellSelectCommunicatorFunction);
}

bool AlienCellFunctionCommunicator::sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageToSend,
                                                                       AlienCell* senderCell,
                                                                       AlienCell* senderPreviousCell,
                                                                       AlienCell* receiverCell) const
{
    AlienCellFunctionCommunicator* communicator = getCommunicator(receiverCell);
    if( communicator ) {
        if( communicator->_receivedMessage.channel == messageToSend.channel ) {
            communicator->_newMessageReceived = true;
            communicator->_receivedMessage.message = messageToSend.message;
            communicator->_receivedMessage.angle = calcAngle(senderCell, senderPreviousCell, receiverCell, grid);
            communicator->_receivedMessage.distance = convertURealToData(_grid->distance(receiverCell, senderCell));
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

