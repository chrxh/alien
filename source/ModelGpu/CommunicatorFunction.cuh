#pragma once

#include "SimulationData.cuh"
#include "Token.cuh"
#include "Cell.cuh"
#include "ConstantMemory.cuh"

class CommunicatorFunction
{
public:
    __inline__ __device__ void init_blockCall(SimulationData* data);
    __inline__ __device__ void processing_blockCall(Token* token);

private:
    __inline__ __device__ static Enums::CommunicatorIn::Type getCommand(Token* token);

    __inline__ __device__ void setListeningChannel(Cell* cell, unsigned char channel) const;
    __inline__ __device__ unsigned char getListeningChannel(Cell* cell) const;
    
    __inline__ __device__ void setAngle(Cell* cell, unsigned char angle) const;
    __inline__ __device__ unsigned char getAngle(Cell* cell) const;

    __inline__ __device__ void setDistance(Cell* cell, unsigned char distance) const;
    __inline__ __device__ unsigned char getDistance(Cell* cell) const;

    __inline__ __device__ void setMessage(Cell* cell, unsigned char message) const;
    __inline__ __device__ unsigned char getMessage(Cell* cell) const;

    __inline__ __device__ void setNewMessageReceived(Cell* cell, bool value) const;
    __inline__ __device__ bool getNewMessageReceived(Cell* cell) const;

    __inline__ __device__ void sendMessage_blockCall(Token* token) const;
    __inline__ __device__ void receiveMessage(Token* token) const;

    struct MessageData {
        unsigned char channel = 0;
        unsigned char message = 0;
        unsigned char angle = 0;
        unsigned char distance = 0;
    };
    __inline__ __device__ void sendMessageToNearbyCommunicators(
        MessageData const& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell, int& numMessages) const;

    __inline__ __device__ bool sendMessageToCommunicatorAndReturnSuccess(
        MessageData const& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell, Cell* receiverCell) const;

    __inline__ __device__ float2 calcDisplacementOfObjectFromSender(
        MessageData const& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;

    __inline__ __device__ unsigned char calcReceivedMessageAngle(Cell* receiverCell, Cell* receiverPreviousCell) const;

private:
    SimulationData* _data;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

namespace {
    struct StaticDataInternal {
        enum Type {
            NewMessageReceived = 0,
            Channel,
            MessageCode,
            OriginAngle,
            OriginDistance,
            _Count
        };
    };
}

__inline__ __device__ void CommunicatorFunction::init_blockCall(SimulationData * data)
{
    _data = data;
}

__inline__ __device__ void CommunicatorFunction::processing_blockCall(Token * token)
{
    __syncthreads();

    auto const command = getCommand(token);
    if (Enums::CommunicatorIn::DO_NOTHING == command) {
        return;
    }

    if (0 == threadIdx.x) {
        token->cell->getLock();
    }
    __syncthreads();

    if (Enums::CommunicatorIn::SET_LISTENING_CHANNEL == command) {
        if (0 == threadIdx.x) {
            setListeningChannel(token->cell, token->memory[Enums::Communicator::IN_CHANNEL]);
        }
    }

    if (Enums::CommunicatorIn::SEND_MESSAGE == command) {
        sendMessage_blockCall(token);
    }

    if (Enums::CommunicatorIn::RECEIVE_MESSAGE == command) {
        if (0 == threadIdx.x) {
            receiveMessage(token);
        }
    }

    __syncthreads();
    if (0 == threadIdx.x) {
        token->cell->releaseLock();
    }
    __syncthreads();
}

__inline__ __device__ Enums::CommunicatorIn::Type CommunicatorFunction::getCommand(Token * token)
{
    return static_cast<Enums::CommunicatorIn::Type>(
        static_cast<unsigned char>(token->memory[Enums::Constr::IN]) % Enums::CommunicatorIn::_COUNTER);
}

__inline__ __device__ void CommunicatorFunction::setListeningChannel(Cell* cell, unsigned char channel) const
{
    cell->staticData[StaticDataInternal::Channel] = channel;
}

__inline__ __device__ unsigned char CommunicatorFunction::getListeningChannel(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::Channel];
}

__inline__ __device__ void CommunicatorFunction::setAngle(Cell * cell, unsigned char angle) const
{
    cell->staticData[StaticDataInternal::OriginAngle] = angle;
}

__inline__ __device__ unsigned char CommunicatorFunction::getAngle(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::OriginAngle];
}

__inline__ __device__ void CommunicatorFunction::setDistance(Cell * cell, unsigned char distance) const
{
    cell->staticData[StaticDataInternal::OriginDistance] = distance;
}

__inline__ __device__ unsigned char CommunicatorFunction::getDistance(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::OriginDistance];
}

__inline__ __device__ void CommunicatorFunction::setMessage(Cell * cell, unsigned char message) const
{
    cell->staticData[StaticDataInternal::MessageCode] = message;
}

__inline__ __device__ unsigned char CommunicatorFunction::getMessage(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::MessageCode];
}

__inline__ __device__ void CommunicatorFunction::setNewMessageReceived(Cell * cell, bool value) const
{
    cell->staticData[StaticDataInternal::NewMessageReceived] = value;
}

__inline__ __device__ bool CommunicatorFunction::getNewMessageReceived(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::NewMessageReceived];
}

__inline__ __device__ void CommunicatorFunction::sendMessage_blockCall(Token * token) const
{
    __shared__ MessageData messageDataToSend;
    if (0 == threadIdx.x) {
        messageDataToSend.channel = token->memory[Enums::Communicator::IN_CHANNEL];
        messageDataToSend.message = token->memory[Enums::Communicator::IN_MESSAGE];
        messageDataToSend.angle = token->memory[Enums::Communicator::IN_ANGLE];
        messageDataToSend.distance = token->memory[Enums::Communicator::IN_DISTANCE];
    }
    __syncthreads();

    __shared__ int numMessages;
    sendMessageToNearbyCommunicators(messageDataToSend, token->cell, token->sourceCell, numMessages);

    __syncthreads();
    if (0 == threadIdx.x) {
        token->memory[Enums::Communicator::OUT_SENT_NUM_MESSAGE] = QuantityConverter::convertIntToData(numMessages);
    }
    __syncthreads();
}

__inline__ __device__ void CommunicatorFunction::receiveMessage(Token * token) const
{
    auto const& cell = token->cell;
    if (getNewMessageReceived(cell)) {
        setNewMessageReceived(cell, false);
        auto const receivedMessageAngle = calcReceivedMessageAngle(cell, token->sourceCell);
        setAngle(cell, receivedMessageAngle);

        token->memory[Enums::Communicator::OUT_RECEIVED_ANGLE] = receivedMessageAngle;
        token->memory[Enums::Communicator::OUT_RECEIVED_DISTANCE] = getDistance(cell);
        token->memory[Enums::Communicator::OUT_RECEIVED_MESSAGE] = getMessage(cell);
        token->memory[Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE] = Enums::CommunicatorOutReceivedNewMessage::YES;
    }
    else {
        token->memory[Enums::Communicator::OUT_RECEIVED_NEW_MESSAGE] = Enums::CommunicatorOutReceivedNewMessage::NO;
    }
}

__inline__ __device__ void CommunicatorFunction::sendMessageToNearbyCommunicators(MessageData const & messageDataToSend, 
    Cell * senderCell, Cell * senderPreviousCell, int & numMessages) const
{
    __shared__ List<Cluster*> clusterList;
    _data->cellFunctionData.mapSectionCollector.getClusters__blockCall(senderCell->absPos,
        cudaSimulationParameters.cellFunctionCommunicatorRange, _data->cellMap, &_data->dynamicMemory, clusterList);

    if (0 == threadIdx.x) {
        numMessages = 0;
    }
    __syncthreads();

    auto const clusters = clusterList.asArray(&_data->dynamicMemory);
    auto const clusterPartition = calcPartition(clusterList.getSize(), threadIdx.x, blockDim.x);

    for (auto clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        auto const& cluster = clusters[clusterIndex];
        for (auto cellIndex = 0; cellIndex < cluster->numCellPointers; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];
            if (cell == senderCell) {
                continue;
            }
            if (Enums::CellFunction::COMMUNICATOR != cell->getCellFunctionType()) {
                continue;
            }
            if (sendMessageToCommunicatorAndReturnSuccess(messageDataToSend, senderCell, senderPreviousCell, cell)) {
                atomicAdd_block(&numMessages, 1);
            }
        }
    }
}

__inline__ __device__ bool CommunicatorFunction::sendMessageToCommunicatorAndReturnSuccess(
    MessageData const& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell, Cell* receiverCell) const
{
    receiverCell->getLock();
    if (getListeningChannel(receiverCell) != messageDataToSend.channel) {
        receiverCell->releaseLock();
        return false;
    }

    auto const displacementOfObjectFromSender = calcDisplacementOfObjectFromSender(messageDataToSend, senderCell, senderPreviousCell);
    auto displacementOfObjectFromReceiver = senderCell->absPos + displacementOfObjectFromSender - receiverCell->absPos;
    _data->cellMap.mapDisplacementCorrection(displacementOfObjectFromReceiver);
    auto const angleSeenFromReceiver = Math::angleOfVector(displacementOfObjectFromReceiver);
    auto const distanceSeenFromReceiver = Math::length(displacementOfObjectFromReceiver);

    setAngle(receiverCell, QuantityConverter::convertAngleToData(angleSeenFromReceiver));
    setDistance(receiverCell, QuantityConverter::convertURealToData(distanceSeenFromReceiver));
    setMessage(receiverCell, messageDataToSend.message);
    setNewMessageReceived(receiverCell, true);

    receiverCell->releaseLock();
    return true;
}

__inline__ __device__ float2 CommunicatorFunction::calcDisplacementOfObjectFromSender(
    MessageData const& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const
{
    auto displacementFromSender = senderPreviousCell->absPos - senderCell->absPos;
    Math::normalize(displacementFromSender);
    displacementFromSender = Math::rotateClockwise(
        displacementFromSender, QuantityConverter::convertDataToAngle(messageDataToSend.angle));
    displacementFromSender = displacementFromSender * QuantityConverter::convertDataToUReal(messageDataToSend.distance);
    return displacementFromSender;
}

__inline__ __device__ unsigned char CommunicatorFunction::calcReceivedMessageAngle(Cell * receiverCell, Cell * receiverPreviousCell) const
{
    auto const displacement = receiverPreviousCell->absPos - receiverCell->absPos;
    auto const localAngle = Math::angleOfVector(displacement);
    auto const messageAngle = QuantityConverter::convertDataToAngle(getAngle(receiverCell));
    auto const relAngle = Math::subtractAngle(messageAngle, localAngle);
    return QuantityConverter::convertAngleToData(relAngle);
}
