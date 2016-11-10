#ifndef CELLFUNCTIONCOMMUNICATOR_H
#define CELLFUNCTIONCOMMUNICATOR_H

#include "model/features/cellfunction.h"

#include <QVector3D>

class TestCellFunctionCommunicator;

class CellFunctionCommunicator : public CellFunction
{
public:
    CellFunctionCommunicator (Grid*& grid);
    CellFunctionCommunicator (quint8* cellFunctionData, Grid*& grid);
    CellFunctionCommunicator (QDataStream& stream, Grid*& grid);

    CellFunctionType getType () const { return CellFunctionType::COMMUNICATOR; }
    void getInternalData (quint8* data);

    friend TestCellFunctionCommunicator;

protected:
    ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell);
    void serializeImpl (QDataStream& stream) const;

private:
    struct MessageData {
        quint8 channel = 0;
        quint8 message = 0;
        quint8 angle = 0;
        quint8 distance = 0;
    };

    COMMUNICATOR_IN readCommandFromToken (Token* token) const;
    void setListeningChannel (Token* token);

    void sendMessageToNearbyCommunicatorsAndUpdateToken (Token* token, Cell* cell, Cell* previousCell) const;
    int sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;
    QList< Cell* > findNearbyCommunicator (Cell* cell) const;
    bool sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell, Cell* receiverCell) const;
    QVector3D calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend, Cell* senderCell, Cell* senderPreviousCell) const;

    void receiveMessage (Token* token,Cell* receiverCell, Cell* receiverPreviousCell);
    void calcReceivedMessageAngle (Cell* receiverCell, Cell* receiverPreviousCell);

    bool _newMessageReceived = false;
    MessageData _receivedMessage;
};

#endif // CELLFUNCTIONCOMMUNICATOR_H
