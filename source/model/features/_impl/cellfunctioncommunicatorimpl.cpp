#include <QString>

#include "model/entities/cell.h"
#include "model/entities/cellcluster.h"
#include "model/entities/token.h"
#include "model/simulationunitcontext.h"
#include "model/cellmap.h"
#include "model/topology.h"
#include "model/physics/physics.h"
#include "model/physics/codingphysicalquantities.h"
#include "model/simulationparameters.h"

#include "cellfunctioncommunicatorimpl.h"

CellFunctionCommunicatorImpl::CellFunctionCommunicatorImpl(SimulationUnitContext* context)
    : CellFunction(context), _parameters(context->getSimulationParameters())
{

}

CellFunctionCommunicatorImpl::CellFunctionCommunicatorImpl (QByteArray data, SimulationUnitContext* context)
	: CellFunction(context), _parameters(context->getSimulationParameters())
{
    _newMessageReceived = static_cast<bool>(data[0]);
    _receivedMessage.channel = data[1];
    _receivedMessage.message = data[2];
    _receivedMessage.angle = data[3];
    _receivedMessage.distance = data[4];
}

bool & CellFunctionCommunicatorImpl::getNewMessageReceivedRef()
{
	return _newMessageReceived;
}

CellFunctionCommunicatorImpl::MessageData & CellFunctionCommunicatorImpl::getReceivedMessageRef()
{
	return _receivedMessage;
}

CellFeature::ProcessingResult CellFunctionCommunicatorImpl::processImpl (Token* token, Cell* cell, Cell* previousCell)
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

void CellFunctionCommunicatorImpl::serializePrimitives (QDataStream& stream) const
{
    stream << _newMessageReceived
           << _receivedMessage.channel
           << _receivedMessage.message
           << _receivedMessage.angle
           << _receivedMessage.distance;
}

void CellFunctionCommunicatorImpl::deserializePrimitives (QDataStream& stream)
{
    stream >> _newMessageReceived
           >> _receivedMessage.channel
           >> _receivedMessage.message
           >> _receivedMessage.angle
           >> _receivedMessage.distance;
}



QByteArray CellFunctionCommunicatorImpl::getInternalData () const
{
	QByteArray data(5, 0);
    data[0] = static_cast<quint8>(_newMessageReceived);
    data[1] = _receivedMessage.channel;
    data[2] = _receivedMessage.message;
    data[3] = _receivedMessage.angle;
    data[4] = _receivedMessage.distance;
	return data;
}

Enums::CommunicatorIn::Type CellFunctionCommunicatorImpl::readCommandFromToken (Token* token) const
{
    return static_cast<Enums::CommunicatorIn::Type>(token->getMemoryRef()[Enums::Communicator::IN] % 4);
}

void CellFunctionCommunicatorImpl::setListeningChannel (Token* token)
{
    _receivedMessage.channel = token->getMemoryRef()[Enums::Communicator::IN_CHANNEL];
}


void CellFunctionCommunicatorImpl::sendMessageToNearbyCommunicatorsAndUpdateToken (Token* token, Cell* cell, Cell* previousCell) const
{
    MessageData messageDataToSend;
    messageDataToSend.channel = token->getMemoryRef()[Enums::Communicator::IN_CHANNEL];
    messageDataToSend.message = token->getMemoryRef()[Enums::Communicator::IN_MESSAGE];
    messageDataToSend.angle = token->getMemoryRef()[Enums::Communicator::IN_ANGLE];
    messageDataToSend.distance = token->getMemoryRef()[Enums::Communicator::IN_DISTANCE];
    int numMsg = sendMessageToNearbyCommunicatorsAndReturnNumber(messageDataToSend, cell, previousCell);
    token->getMemoryRef()[Enums::Communicator::OUT_SENT_NUM_MESSAGE] = CodingPhysicalQuantities::convertIntToData(numMsg);
}

int CellFunctionCommunicatorImpl::sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend,
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

QList< Cell* > CellFunctionCommunicatorImpl::findNearbyCommunicator(Cell* cell) const
{
    CellMap::CellSelectFunction cellSelectCommunicatorFunction =
        [](Cell* cell)
        {
            CellFunction* cellFunction = cell->getFeatures()->findObject<CellFunction>();
            return cellFunction && (cellFunction->getType() == Enums::CellFunction::COMMUNICATOR);
        };
    QVector3D cellPos = cell->calcPosition();
    qreal range = _parameters->cellFunctionCommunicatorRange;
    return _context->getCellMap()->getNearbySpecificCells(cellPos, range, cellSelectCommunicatorFunction);
}

bool CellFunctionCommunicatorImpl::sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend,
                                                                       Cell* senderCell,
                                                                       Cell* senderPreviousCell,
                                                                       Cell* receiverCell) const
{
    CellFunctionCommunicatorImpl* communicator = receiverCell->getFeatures()->findObject<CellFunctionCommunicatorImpl>();
    if( communicator ) {
        if( communicator->_receivedMessage.channel == messageDataToSend.channel ) {
            QVector3D displacementOfObjectFromSender = calcDisplacementOfObjectFromSender(messageDataToSend, senderCell, senderPreviousCell);
            Topology* topology = _context->getTopology();
            QVector3D displacementOfObjectFromReceiver = topology->displacement(receiverCell->calcPosition(), senderCell->calcPosition() + displacementOfObjectFromSender);
            qreal angleSeenFromReceiver = Physics::angleOfVector(displacementOfObjectFromReceiver);
            qreal distanceSeenFromReceiver = displacementOfObjectFromReceiver.length();
            communicator->_receivedMessage.angle = CodingPhysicalQuantities::convertAngleToData(angleSeenFromReceiver);
            communicator->_receivedMessage.distance = CodingPhysicalQuantities::convertURealToData(distanceSeenFromReceiver);
            communicator->_receivedMessage.message = messageDataToSend.message;
            communicator->_newMessageReceived = true;
            return true;
        }
    }
    return false;
}

QVector3D CellFunctionCommunicatorImpl::calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend
	, Cell* senderCell, Cell* senderPreviousCell) const
{
    QVector3D displacementFromSender = senderPreviousCell->calcPosition() - senderCell->calcPosition();
    displacementFromSender.normalize();
    displacementFromSender = Physics::rotateClockwise(displacementFromSender, CodingPhysicalQuantities::convertDataToAngle(messageDataToSend.angle));
    displacementFromSender = displacementFromSender * CodingPhysicalQuantities::convertDataToUReal(messageDataToSend.distance);
    return displacementFromSender;
}

void CellFunctionCommunicatorImpl::receiveMessage (Token* token, Cell* receiverCell, Cell* receiverPreviousCell)
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

void CellFunctionCommunicatorImpl::calcReceivedMessageAngle (Cell* receiverCell,
                                                              Cell* receiverPreviousCell)
{
    QVector3D displacement = receiverPreviousCell->calcPosition() - receiverCell->calcPosition();
    qreal localAngle = Physics::angleOfVector(displacement);
    qreal messageAngle = CodingPhysicalQuantities::convertDataToAngle(_receivedMessage.angle);
    qreal relAngle = Physics::subtractAngle(messageAngle, localAngle);
    _receivedMessage.angle = CodingPhysicalQuantities::convertAngleToData(relAngle);
}
