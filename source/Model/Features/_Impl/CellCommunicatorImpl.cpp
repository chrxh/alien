#include <QString>

#include "Model/Entities/Cell.h"
#include "Model/Entities/Cluster.h"
#include "Model/Entities/Token.h"
#include "Model/Context/UnitContext.h"
#include "Model/Context/CellMap.h"
#include "Model/Context/SpaceMetricLocal.h"
#include "Model/Physics/Physics.h"
#include "Model/Physics/PhysicalQuantityConverter.h"
#include "Model/SimulationParameters.h"

#include "CellCommunicatorImpl.h"

CellCommunicatorImpl::CellCommunicatorImpl(UnitContext* context)
    : CellFunction(context), _parameters(context->getSimulationParameters())
{

}

CellCommunicatorImpl::CellCommunicatorImpl (QByteArray data, UnitContext* context)
	: CellFunction(context), _parameters(context->getSimulationParameters())
{
    _newMessageReceived = static_cast<bool>(data[0]);
    _receivedMessage.channel = data[1];
    _receivedMessage.message = data[2];
    _receivedMessage.angle = data[3];
    _receivedMessage.distance = data[4];
}

bool & CellCommunicatorImpl::getNewMessageReceivedRef()
{
	return _newMessageReceived;
}

CellCommunicatorImpl::MessageData & CellCommunicatorImpl::getReceivedMessageRef()
{
	return _receivedMessage;
}

CellFeature::ProcessingResult CellCommunicatorImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
{
    ProcessingResult processingResult {false, 0};
    Enums::CommunicatorIn::Type cmd = readCommandFromToken(token);
    if( cmd == Enums::CommunicatorIn::SET_LISTENING_CHANNEL )
        setListeningChannel(token);
    if( cmd == Enums::CommunicatorIn::SEND_MESSAGE )
        sendMessageToNearbyCommunicatorsAndUpdateToken(token, cell, previousCell);
    if( cmd == Enums::CommunicatorIn::RECEIVE_MESSAGE )
        receiveMessage(token, cell, previousCell);
    return processingResult;
}

void CellCommunicatorImpl::serializePrimitives (QDataStream& stream) const
{
    stream << _newMessageReceived
           << _receivedMessage.channel
           << _receivedMessage.message
           << _receivedMessage.angle
           << _receivedMessage.distance;
}

void CellCommunicatorImpl::deserializePrimitives (QDataStream& stream)
{
    stream >> _newMessageReceived
           >> _receivedMessage.channel
           >> _receivedMessage.message
           >> _receivedMessage.angle
           >> _receivedMessage.distance;
}



QByteArray CellCommunicatorImpl::getInternalData () const
{
	QByteArray data(5, 0);
    data[0] = static_cast<quint8>(_newMessageReceived);
    data[1] = _receivedMessage.channel;
    data[2] = _receivedMessage.message;
    data[3] = _receivedMessage.angle;
    data[4] = _receivedMessage.distance;
	return data;
}

Enums::CommunicatorIn::Type CellCommunicatorImpl::readCommandFromToken (Token* token) const
{
    return static_cast<Enums::CommunicatorIn::Type>(token->getMemoryRef()[Enums::Communicator::IN] % 4);
}

void CellCommunicatorImpl::setListeningChannel (Token* token)
{
    _receivedMessage.channel = token->getMemoryRef()[Enums::Communicator::IN_CHANNEL];
}


void CellCommunicatorImpl::sendMessageToNearbyCommunicatorsAndUpdateToken (Token* token, Cell* cell, Cell* previousCell) const
{
    MessageData messageDataToSend;
    messageDataToSend.channel = token->getMemoryRef()[Enums::Communicator::IN_CHANNEL];
    messageDataToSend.message = token->getMemoryRef()[Enums::Communicator::IN_MESSAGE];
    messageDataToSend.angle = token->getMemoryRef()[Enums::Communicator::IN_ANGLE];
    messageDataToSend.distance = token->getMemoryRef()[Enums::Communicator::IN_DISTANCE];
    int numMsg = sendMessageToNearbyCommunicatorsAndReturnNumber(messageDataToSend, cell, previousCell);
    token->getMemoryRef()[Enums::Communicator::OUT_SENT_NUM_MESSAGE] = PhysicalQuantityConverter::convertIntToData(numMsg);
}

int CellCommunicatorImpl::sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend,
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

QList< Cell* > CellCommunicatorImpl::findNearbyCommunicator(Cell* cell) const
{
    CellMap::CellSelectFunction cellSelectCommunicatorFunction =
        [](Cell* cell)
        {
            CellFunction* cellFunction = cell->getFeatures()->findObject<CellFunction>();
            return cellFunction && (cellFunction->getType() == Enums::CellFunction::COMMUNICATOR);
        };
    QVector2D cellPos = cell->calcPosition();
    qreal range = _parameters->cellFunctionCommunicatorRange;
    return _context->getCellMap()->getNearbySpecificCells(cellPos, range, cellSelectCommunicatorFunction);
}

bool CellCommunicatorImpl::sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend,
                                                                       Cell* senderCell,
                                                                       Cell* senderPreviousCell,
                                                                       Cell* receiverCell) const
{
    CellCommunicatorImpl* communicator = receiverCell->getFeatures()->findObject<CellCommunicatorImpl>();
    if( communicator ) {
        if( communicator->_receivedMessage.channel == messageDataToSend.channel ) {
            QVector2D displacementOfObjectFromSender = calcDisplacementOfObjectFromSender(messageDataToSend, senderCell, senderPreviousCell);
            SpaceMetricLocal* metric = _context->getSpaceMetric();
            QVector2D displacementOfObjectFromReceiver = metric->displacement(receiverCell->calcPosition(), senderCell->calcPosition() + displacementOfObjectFromSender);
            qreal angleSeenFromReceiver = Physics::angleOfVector(displacementOfObjectFromReceiver);
            qreal distanceSeenFromReceiver = displacementOfObjectFromReceiver.length();
            communicator->_receivedMessage.angle = PhysicalQuantityConverter::convertAngleToData(angleSeenFromReceiver);
            communicator->_receivedMessage.distance = PhysicalQuantityConverter::convertURealToData(distanceSeenFromReceiver);
            communicator->_receivedMessage.message = messageDataToSend.message;
            communicator->_newMessageReceived = true;
            return true;
        }
    }
    return false;
}

QVector2D CellCommunicatorImpl::calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend
	, Cell* senderCell, Cell* senderPreviousCell) const
{
    QVector2D displacementFromSender = senderPreviousCell->calcPosition() - senderCell->calcPosition();
    displacementFromSender.normalize();
    displacementFromSender = Physics::rotateClockwise(displacementFromSender, PhysicalQuantityConverter::convertDataToAngle(messageDataToSend.angle));
    displacementFromSender = displacementFromSender * PhysicalQuantityConverter::convertDataToUReal(messageDataToSend.distance);
    return displacementFromSender;
}

void CellCommunicatorImpl::receiveMessage (Token* token, Cell* receiverCell, Cell* receiverPreviousCell)
{
	QByteArray& tokenMem = token->getMemoryRef();
	if (_newMessageReceived) {
        _newMessageReceived = false;
        calcReceivedMessageAngle(receiverCell, receiverPreviousCell);
		tokenMem[Enums::Communicator::OUT_RECEIVED_ANGLE] = _receivedMessage.angle;
		tokenMem[Enums::Communicator::OUT_RECEIVED_DISTANCE] = _receivedMessage.distance;
		tokenMem[Enums::Communicator::OUT_RECEIVED_MESSAGE] = _receivedMessage.message;
		tokenMem[Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE] = Enums::CommunicatorOutReceivedNewMessage::YES;
    }
	else {
		tokenMem[Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE]
			= Enums::CommunicatorOutReceivedNewMessage::NO;
	}
}

void CellCommunicatorImpl::calcReceivedMessageAngle (Cell* receiverCell,
                                                              Cell* receiverPreviousCell)
{
    QVector2D displacement = receiverPreviousCell->calcPosition() - receiverCell->calcPosition();
    qreal localAngle = Physics::angleOfVector(displacement);
    qreal messageAngle = PhysicalQuantityConverter::convertDataToAngle(_receivedMessage.angle);
    qreal relAngle = Physics::subtractAngle(messageAngle, localAngle);
    _receivedMessage.angle = PhysicalQuantityConverter::convertAngleToData(relAngle);
}
