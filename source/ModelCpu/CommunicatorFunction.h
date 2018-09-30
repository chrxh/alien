#pragma once

#include "Model/Local/CellFunction.h"

#include <QVector2D>

class CommunicatorFunction
	: public CellFunction
{
public:
    CommunicatorFunction (UnitContext* context);
    CommunicatorFunction (QByteArray data, UnitContext* context);

    Enums::CellFunction::Type getType () const { return Enums::CellFunction::COMMUNICATOR; }
	
	struct InternalDataSemantic {
		enum Type {
			NewMessageReceived = 0,
			Channel,
			MessageCode,
			OriginAngle,
			OriginDistance,
			_Count
		};
	};
	QByteArray getInternalData () const override;

	struct MessageData {
		quint8 channel = 0;
		quint8 message = 0;
		quint8 angle = 0;
		quint8 distance = 0;
	};

protected:
	virtual ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) override;

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
