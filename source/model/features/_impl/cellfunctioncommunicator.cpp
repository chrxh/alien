#include "cellfunctioncommunicator.h"
#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/physics/physics.h"

#include <QString>

CellFunctionCommunicator::CellFunctionCommunicator(Grid*& grid)
    : CellFunction(grid)
{

}

CellFunctionCommunicator::CellFunctionCommunicator (quint8* cellFunctionData, Grid*& grid)
    : CellFunction(grid)
{
    _newMessageReceived = static_cast<bool>(cellFunctionData[0]);
    _receivedMessage.channel = cellFunctionData[1];
    _receivedMessage.message = cellFunctionData[2];
    _receivedMessage.angle = cellFunctionData[3];
    _receivedMessage.distance = cellFunctionData[4];
}

CellFunctionCommunicator::CellFunctionCommunicator (QDataStream& stream, Grid*& grid)
    : CellFunction(grid)
{
    stream >> _newMessageReceived
           >> _receivedMessage.channel
           >> _receivedMessage.message
           >> _receivedMessage.angle
           >> _receivedMessage.distance;
}

CellFeature::ProcessingResult CellFunctionCommunicator::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    COMMUNICATOR_IN cmd = readCommandFromToken(token);
    if( cmd == COMMUNICATOR_IN::SET_LISTENING_CHANNEL )
        setListeningChannel(token);
    if( cmd == COMMUNICATOR_IN::SEND_MESSAGE )
        sendMessageToNearbyCommunicatorsAndUpdateToken(token, cell, previousCell);
    if( cmd == COMMUNICATOR_IN::RECEIVE_MESSAGE )
        receiveMessage(token, cell, previousCell);
    return processingResult;
}

void CellFunctionCommunicator::serialize (QDataStream& stream) const
{
    stream << _newMessageReceived
           << _receivedMessage.channel
           << _receivedMessage.message
           << _receivedMessage.angle
           << _receivedMessage.distance;
}


void CellFunctionCommunicator::getInternalData (quint8* data) const
{
    data[0] = static_cast<quint8>(_newMessageReceived);
    data[1] = _receivedMessage.channel;
    data[2] = _receivedMessage.message;
    data[3] = _receivedMessage.angle;
    data[4] = _receivedMessage.distance;
}

COMMUNICATOR_IN CellFunctionCommunicator::readCommandFromToken (Token* token) const
{
    return static_cast<COMMUNICATOR_IN>(token->memory[static_cast<int>(COMMUNICATOR::IN)] % 4);
}

void CellFunctionCommunicator::setListeningChannel (Token* token)
{
    _receivedMessage.channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
}


void CellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndUpdateToken (Token* token,
                                                                            Cell* cell,
                                                                            Cell* previousCell) const
{
    MessageData messageDataToSend;
    messageDataToSend.channel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
    messageDataToSend.message = token->memory[static_cast<int>(COMMUNICATOR::IN_MESSAGE)];
    messageDataToSend.angle = token->memory[static_cast<int>(COMMUNICATOR::IN_ANGLE)];
    messageDataToSend.distance = token->memory[static_cast<int>(COMMUNICATOR::IN_DISTANCE)];
    int numMsg = sendMessageToNearbyCommunicatorsAndReturnNumber(messageDataToSend, cell, previousCell);
    token->memory[static_cast<int>(COMMUNICATOR::OUT_SENT_NUM_MESSAGE)] = convertIntToData(numMsg);
}

int CellFunctionCommunicator::sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend,
                                                                            Cell* senderCell,
                                                                            Cell* senderPreviousCell) const
{
    int numMsg = 0;
    QList< Cell* > nearbyCommunicatorCells = findNearbyCommunicator (senderCell);
    foreach(Cell* nearbyCell, nearbyCommunicatorCells)
        if( nearbyCell != senderCell )
            if( sendMessageToCommunicatorAndReturnSuccess(messageDataToSend, senderCell, senderPreviousCell, nearbyCell) )
                ++numMsg;
    return numMsg;
}

QList< Cell* > CellFunctionCommunicator::findNearbyCommunicator(Cell* cell) const
{
    Grid::CellSelectFunction cellSelectCommunicatorFunction =
        [](Cell* cell)
        {
            CellFunction* cellFunction = CellFeature::findObject<CellFunction>(cell->getFeatures());
            return cellFunction && (cellFunction->getType() == CellFunctionType::COMMUNICATOR);
        };
    QVector3D cellPos = cell->calcPosition();
    qreal range = simulationParameters.CELL_FUNCTION_COMMUNICATOR_RANGE;
    return _grid->getNearbySpecificCells(cellPos, range, cellSelectCommunicatorFunction);
}

bool CellFunctionCommunicator::sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend,
                                                                       Cell* senderCell,
                                                                       Cell* senderPreviousCell,
                                                                       Cell* receiverCell) const
{
    CellFunctionCommunicator* communicator = CellFeature::findObject<CellFunctionCommunicator>(receiverCell->getFeatures());
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

QVector3D CellFunctionCommunicator::calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend,
                                                                             Cell* senderCell,
                                                                             Cell* senderPreviousCell) const
{
    QVector3D displacementFromSender = senderPreviousCell->calcPosition() - senderCell->calcPosition();
    displacementFromSender.normalize();
    displacementFromSender = Physics::rotateClockwise(displacementFromSender, convertDataToAngle(messageDataToSend.angle));
    displacementFromSender = displacementFromSender*convertDataToUReal(messageDataToSend.distance);
    return displacementFromSender;
}

void CellFunctionCommunicator::receiveMessage (Token* token,
                                                    Cell* receiverCell,
                                                    Cell* receiverPreviousCell)
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

void CellFunctionCommunicator::calcReceivedMessageAngle (Cell* receiverCell,
                                                              Cell* receiverPreviousCell)
{
    QVector3D displacement = receiverPreviousCell->calcPosition() - receiverCell->calcPosition();
    qreal localAngle = Physics::angleOfVector(displacement);
    qreal messageAngle = convertDataToAngle(_receivedMessage.angle);
    qreal relAngle = Physics::subtractAngle(messageAngle, localAngle);
    _receivedMessage.angle = convertAngleToData(relAngle);
}
