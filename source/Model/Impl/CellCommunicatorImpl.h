#ifndef CELLFUNCTIONCOMMUNICATORIMPL_H
#define CELLFUNCTIONCOMMUNICATORIMPL_H

#include "Model/Local/CellFunction.h"

#include <QVector2D>

class CellFunctionCommunicatorTest;

class CellCommunicatorImpl
	: public CellFunction
{
public:
    CellCommunicatorImpl (UnitContext* context);
    CellCommunicatorImpl (QByteArray data, UnitContext* context);

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives (QDataStream& stream) override;

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::COMMUNICATOR; }
	QByteArray getInternalData () const override;

	struct MessageData {
		quint8 channel = 0;
		quint8 message = 0;
		quint8 angle = 0;
		quint8 distance = 0;
	};
    bool& getNewMessageReceivedRef();
    MessageData& getReceivedMessageRef();

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) override;

private:

	SimulationParameters* _parameters = nullptr;

    bool _newMessageReceived = false;
    MessageData _receivedMessage;

    Enums::CommunicatorIn::Type readCommandFromToken (Token* token) const;
    void setListeningChannel (Token* token);

    void sendMessageToNearbyCommunicatorsAndUpdateToken (Token* token, Cell* cell, Cell* previousCell) const;
    int sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;
    QList< Cell* > findNearbyCommunicator (Cell* cell) const;
    bool sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell, Cell* receiverCell) const;
    QVector2D calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;

    void receiveMessage (Token* token,Cell* receiverCell, Cell* receiverPreviousCell);
    void calcReceivedMessageAngle (Cell* receiverCell, Cell* receiverPreviousCell);
};

#endif // CELLFUNCTIONCOMMUNICATORIMPL_H
