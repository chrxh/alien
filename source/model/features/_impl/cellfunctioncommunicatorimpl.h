#ifndef CELLFUNCTIONCOMMUNICATORIMPL_H
#define CELLFUNCTIONCOMMUNICATORIMPL_H

#include "model/features/cellfunction.h"

#include <QVector3D>

class TestCellFunctionCommunicator;

class CellFunctionCommunicatorImpl : public CellFunction
{
public:
    CellFunctionCommunicatorImpl (SimulationContext* context);
    CellFunctionCommunicatorImpl (quint8* cellFunctionData, SimulationContext* context);

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives (QDataStream& stream) const override;

    CellFunctionType getType () const { return CellFunctionType::COMMUNICATOR; }
    void getInternalData (quint8* ptr) const override;

	struct MessageData {
		quint8 channel = 0;
		quint8 message = 0;
		quint8 angle = 0;
		quint8 distance = 0;
	};
    bool& getNewMessageReceivedRef() override;
    MessageData& getReceivedMessageRef() override;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:

    bool _newMessageReceived = false;
    MessageData _receivedMessage;

    COMMUNICATOR_IN readCommandFromToken (Token* token) const;
    void setListeningChannel (Token* token);

    void sendMessageToNearbyCommunicatorsAndUpdateToken (Token* token, Cell* cell, Cell* previousCell) const;
    int sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;
    QList< Cell* > findNearbyCommunicator (Cell* cell) const;
    bool sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell, Cell* receiverCell) const;
    QVector3D calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;

    void receiveMessage (Token* token,Cell* receiverCell, Cell* receiverPreviousCell);
    void calcReceivedMessageAngle (Cell* receiverCell, Cell* receiverPreviousCell);
};

#endif // CELLFUNCTIONCOMMUNICATORIMPL_H
