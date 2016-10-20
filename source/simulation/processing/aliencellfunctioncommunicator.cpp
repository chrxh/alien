#include "aliencellfunctioncommunicator.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"
#include "../physics/physics.h"

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
    stream >> _newMessageReceived
           >> _receivedMessage.channel
           >> _receivedMessage.message
           >> _receivedMessage.angle
           >> _receivedMessage.distance;
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
        receiveMessage(token, cell, previousCell);
}

QString AlienCellFunctionCommunicator::getCellFunctionName () const
{
    return "COMMUNICATOR";
}

void AlienCellFunctionCommunicator::serialize (QDataStream& stream)
{
    AlienCellFunction::serialize(stream);
    stream << _newMessageReceived
           << _receivedMessage.channel
           << _receivedMessage.message
           << _receivedMessage.angle
           << _receivedMessage.distance;
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
    MessageData messageDataToSend;
    messageDataToSend.channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
    messageDataToSend.message = token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)];
    messageDataToSend.angle = token->memory[static_cast<int>(COMMUNICATOR::IN_ANGLE)];
    messageDataToSend.distance = token->memory[static_cast<int>(COMMUNICATOR::IN_DISTANCE)];
    int numMsg = sendMessageToNearbyCommunicatorsAndReturnNumber(messageDataToSend, cell, previousCell);
    token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)] = convertIntToData(numMsg);
}

int AlienCellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend,
                                                                            AlienCell* senderCell,
                                                                            AlienCell* senderPreviousCell) const
{
    int numMsg = 0;
    QList< AlienCell* > nearbyCommunicatorCells = findNearbyCommunicator (senderCell);
    foreach(AlienCell* nearbyCell, nearbyCommunicatorCells)
        if( nearbyCell != senderCell )
            if( sendMessageToCommunicatorAndReturnSuccess(messageDataToSend, senderCell, senderPreviousCell, nearbyCell) )
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

bool AlienCellFunctionCommunicator::sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend,
                                                                       AlienCell* senderCell,
                                                                       AlienCell* senderPreviousCell,
                                                                       AlienCell* receiverCell) const
{
    AlienCellFunctionCommunicator* communicator = getCommunicator(receiverCell);
    if( communicator ) {
        if( communicator->_receivedMessage.channel == messageDataToSend.channel ) {
            QVector3D displacementOfObjectFromSender = calcDisplacementOfObjectFromSender(messageDataToSend, senderCell, senderPreviousCell);
            QVector3D displacementOfObjectFromReceiver = _grid->displacement(receiverCell->calcPosition(), senderCell->calcPosition() + displacementOfObjectFromSender);
            qreal angleSeenFromReceiver = Physics::angleOfVector(displacementOfObjectFromReceiver);
            qreal distanceSeenFromReceiver = displacementOfObjectFromReceiver.length();
            communicator->_receivedMessage.angle = convertAngleToData(angleSeenFromReceiver);
            communicator->_receivedMessage.distance = convertURealToData(distanceSeenFromReceiver);
            communicator->_receivedMessage.message = messageDataToSend.message;
            communicator->_newMessageReceived = true;
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

QVector3D AlienCellFunctionCommunicator::calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend,
                                                                             AlienCell* senderCell,
                                                                             AlienCell* senderPreviousCell) const
{
    QVector3D displacementFromSender = senderPreviousCell->calcPosition() - senderCell->calcPosition();
    displacementFromSender.normalize();
    displacementFromSender = Physics::rotateClockwise(displacementFromSender, convertDataToAngle(messageDataToSend.angle));
    displacementFromSender = displacementFromSender*convertDataToUReal(messageDataToSend.distance);
    return displacementFromSender;
}

void AlienCellFunctionCommunicator::receiveMessage (AlienToken* token,
                                                    AlienCell* receiverCell,
                                                    AlienCell* receiverPreviousCell)
{
    if( _newMessageReceived ) {
        _newMessageReceived = false;
        calcReceivedMessageAngle(receiverCell, receiverPreviousCell);
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_ANGLE)]
                = _receivedMessage.angle;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_DISTANCE)]
                = _receivedMessage.distance;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_MESSAGE)]
                = _receivedMessage.message;
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NEW_MESSAGE);
    }
    else
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO_NEW_MESSAGE);
}

void AlienCellFunctionCommunicator::calcReceivedMessageAngle (AlienCell* receiverCell,
                                                              AlienCell* receiverPreviousCell)
{
    QVector3D displacement = receiverPreviousCell->calcPosition() - receiverCell->calcPosition();
    qreal localAngle = Physics::angleOfVector(displacement);
    qreal messageAngle = convertDataToAngle(_receivedMessage.angle);
    qreal relAngle = Physics::subtractAngle(messageAngle, localAngle);
    _receivedMessage.angle = convertAngleToData(relAngle);
}
