#pragma once

#include "SimulationData.cuh"
#include "Token.cuh"
#include "Cell.cuh"
#include "ConstantMemory.cuh"

class CommunicatorProcessor
{
public:
    __inline__ __device__ void init_block(SimulationData* data);
    __inline__ __device__ void processing_block(Token* token);

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

    __inline__ __device__ void sendMessage_block(Token* token) const;
    __inline__ __device__ void receiveMessage(Token* token) const;

    struct MessageData {
        unsigned char channel;
        unsigned char message;
        unsigned char angle;
        unsigned char distance;
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

__inline__ __device__ void CommunicatorProcessor::init_block(SimulationData * data)
{
    _data = data;
}

__inline__ __device__ void CommunicatorProcessor::processing_block(Token * token)
{
    __syncthreads();

    __shared__ Enums::CommunicatorIn::Type command;
    if (0 == threadIdx.x) {
        command = getCommand(token);
    }
    __syncthreads();

    if (Enums::CommunicatorIn_DoNothing == command) {
        return;
    }

    if (Enums::CommunicatorIn_SetListeningChannel == command) {
        if (0 == threadIdx.x) {
            token->cell->getLock();
            setListeningChannel(token->cell, token->memory[Enums::Communicator_InChannel]);
            token->cell->releaseLock();
        }
    }

    if (Enums::CommunicatorIn_SendMessage == command) {
        sendMessage_block(token);
    }

    if (Enums::CommunicatorIn_ReceiveMessage == command) {
        if (0 == threadIdx.x) {
            token->cell->getLock();
            receiveMessage(token);
            token->cell->releaseLock();
        }
    }
    __syncthreads();
}

__inline__ __device__ Enums::CommunicatorIn::Type CommunicatorProcessor::getCommand(Token * token)
{
    return static_cast<Enums::CommunicatorIn::Type>(
        static_cast<unsigned char>(token->memory[Enums::Communicator_Input]) % Enums::CommunicatorIn::_COUNTER);
}

__inline__ __device__ void CommunicatorProcessor::setListeningChannel(Cell* cell, unsigned char channel) const
{
    cell->staticData[StaticDataInternal::Channel] = channel;
}

__inline__ __device__ unsigned char CommunicatorProcessor::getListeningChannel(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::Channel];
}

__inline__ __device__ void CommunicatorProcessor::setAngle(Cell * cell, unsigned char angle) const
{
    cell->staticData[StaticDataInternal::OriginAngle] = angle;
}

__inline__ __device__ unsigned char CommunicatorProcessor::getAngle(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::OriginAngle];
}

__inline__ __device__ void CommunicatorProcessor::setDistance(Cell * cell, unsigned char distance) const
{
    cell->staticData[StaticDataInternal::OriginDistance] = distance;
}

__inline__ __device__ unsigned char CommunicatorProcessor::getDistance(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::OriginDistance];
}

__inline__ __device__ void CommunicatorProcessor::setMessage(Cell * cell, unsigned char message) const
{
    cell->staticData[StaticDataInternal::MessageCode] = message;
}

__inline__ __device__ unsigned char CommunicatorProcessor::getMessage(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::MessageCode];
}

__inline__ __device__ void CommunicatorProcessor::setNewMessageReceived(Cell * cell, bool value) const
{
    cell->staticData[StaticDataInternal::NewMessageReceived] = value;
}

__inline__ __device__ bool CommunicatorProcessor::getNewMessageReceived(Cell * cell) const
{
    return cell->staticData[StaticDataInternal::NewMessageReceived];
}

__inline__ __device__ void CommunicatorProcessor::sendMessage_block(Token * token) const
{
    __shared__ MessageData messageDataToSend;
    if (0 == threadIdx.x) {
        messageDataToSend.channel = token->memory[Enums::Communicator_InChannel];
        messageDataToSend.message = token->memory[Enums::Communicator_InMessage];
        messageDataToSend.angle = token->memory[Enums::Communicator_InAngle];
        messageDataToSend.distance = token->memory[Enums::Communicator_InDistance];
    }
    __syncthreads();

    __shared__ int numMessages;
    sendMessageToNearbyCommunicators(messageDataToSend, token->cell, token->sourceCell, numMessages);

    __syncthreads();
    if (0 == threadIdx.x) {
        token->memory[Enums::Communicator_OutSentNumMessage] = QuantityConverter::convertIntToData(numMessages);
    }
    __syncthreads();
}

__inline__ __device__ void CommunicatorProcessor::receiveMessage(Token * token) const
{
    auto const& cell = token->cell;
    if (getNewMessageReceived(cell)) {
        setNewMessageReceived(cell, false);
        auto const receivedMessageAngle = calcReceivedMessageAngle(cell, token->sourceCell);
        setAngle(cell, receivedMessageAngle);

        token->memory[Enums::Communicator_OutReceivedAngle] = receivedMessageAngle;
        token->memory[Enums::Communicator_OutReceivedDistance] = getDistance(cell);
        token->memory[Enums::Communicator_OutReceivedMessage] = getMessage(cell);
        token->memory[Enums::Communicator_OutReceivedNewMessage] = Enums::CommunicatorOutReceivedNewMessage_Yes;
    }
    else {
        token->memory[Enums::Communicator_OutReceivedNewMessage] = Enums::CommunicatorOutReceivedNewMessage_No;
    }
}

__inline__ __device__ void CommunicatorProcessor::sendMessageToNearbyCommunicators(MessageData const & messageDataToSend, 
    Cell * senderCell, Cell * senderPreviousCell, int & numMessages) const
{ 
    __shared__ List<Cluster*> clusterList;
    _data->cellFunctionData.mapSectionCollector.getClusters_block(senderCell->absPos,
        cudaSimulationParameters.cellFunctionCommunicatorRange, _data->cellMap, &_data->tempMemory, clusterList);

    __shared__ Cluster** clusters;

    if (0 == threadIdx.x) {
        numMessages = 0;
        clusters = clusterList.asArray(&_data->tempMemory);
    }
    __syncthreads();

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

__inline__ __device__ bool CommunicatorProcessor::sendMessageToCommunicatorAndReturnSuccess(
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

__inline__ __device__ float2 CommunicatorProcessor::calcDisplacementOfObjectFromSender(
    MessageData const& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const
{
    auto displacementFromSender = senderPreviousCell->absPos - senderCell->absPos;
    Math::normalize(displacementFromSender);
    displacementFromSender = Math::rotateClockwise(
        displacementFromSender, QuantityConverter::convertDataToAngle(messageDataToSend.angle));
    displacementFromSender = displacementFromSender * QuantityConverter::convertDataToUReal(messageDataToSend.distance);
    return displacementFromSender;
}

__inline__ __device__ unsigned char CommunicatorProcessor::calcReceivedMessageAngle(Cell * receiverCell, Cell * receiverPreviousCell) const
{
    auto const displacement = receiverPreviousCell->absPos - receiverCell->absPos;
    auto const localAngle = Math::angleOfVector(displacement);
    auto const messageAngle = QuantityConverter::convertDataToAngle(getAngle(receiverCell));
    auto const relAngle = Math::subtractAngle(messageAngle, localAngle);
    return QuantityConverter::convertAngleToData(relAngle);
}
