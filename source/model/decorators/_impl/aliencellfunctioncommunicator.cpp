#include "aliencellfunctioncommunicator.h"
#include "model/entities/aliencell.h"
#include "model/entities/aliencellcluster.h"
#include "model/entities/alientoken.h"
#include "model/physics/physics.h"

#include <QString>

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator(AlienCell* cell, AlienGrid*& grid)
    : AlienCellFunction(cell, grid)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid)
    : AlienCellFunctionCommunicator(cell, grid)
{
    _newMessageReceived = static_cast<bool>(cellFunctionData[0]);
    _receivedMessage.channel = cellFunctionData[1];
    _receivedMessage.message = cellFunctionData[2];
    _receivedMessage.angle = cellFunctionData[3];
    _receivedMessage.distance = cellFunctionData[4];
}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (AlienCell* cell, QDataStream& stream, AlienGrid*& grid)
    : AlienCellFunctionCommunicator(cell, grid)
{
    stream >> _newMessageReceived
           >> _receivedMessage.channel
           >> _receivedMessage.message
           >> _receivedMessage.angle
           >> _receivedMessage.distance;
}

AlienCell::ProcessingResult AlienCellFunctionCommunicator::process (AlienToken* token, AlienCell* previousCell)
{
    AlienCell::ProcessingResult processingResult = _cell->process(token, previousCell);
    COMMUNICATOR_IN cmd = readCommandFromToken(token);
    if( cmd == COMMUNICATOR_IN::SET_LISTENING_CHANNEL )
        setListeningChannel(token);
    if( cmd == COMMUNICATOR_IN::SEND_MESSAGE )
        sendMessageToNearbyCommunicatorsAndUpdateToken(token, _cell, previousCell);
    if( cmd == COMMUNICATOR_IN::RECEIVE_MESSAGE )
        receiveMessage(token, _cell, previousCell);
    return processingResult;
}

void AlienCellFunctionCommunicator::serialize (QDataStream& stream)
{
    AlienCellDecorator::serialize(stream);
    stream << _newMessageReceived
           << _receivedMessage.channel
           << _receivedMessage.message
           << _receivedMessage.angle
           << _receivedMessage.distance;
}


void AlienCellFunctionCommunicator::getInternalData (quint8* data)
{
    data[0] = static_cast<quint8>(_newMessageReceived);
    data[1] = _receivedMessage.channel;
    data[2] = _receivedMessage.message;
    data[3] = _receivedMessage.angle;
    data[4] = _receivedMessage.distance;
}

COMMUNICATOR_IN AlienCellFunctionCommunicator::readCommandFromToken (AlienToken* token) const
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
            AlienCellFunction* cellFunction = AlienCellDecorator::findObject<AlienCellFunction>(cell);
            return cellFunction && (cellFunction->getType() == CellFunctionType::COMMUNICATOR);
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
    AlienCellFunctionCommunicator* communicator = AlienCellDecorator::findObject<AlienCellFunctionCommunicator>(receiverCell);
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
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::YES);
    }
    else
        token->memory[static_cast<int>(COMMUNICATOR::OUT_RECEIVED_NEW_MESSAGE)]
                = static_cast<int>(COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE::NO);
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
