#ifndef ALIENCELLFUNCTIONCOMMUNICATOR_H
#define ALIENCELLFUNCTIONCOMMUNICATOR_H

#include "model/decorators/aliencellfunction.h"

#include <QVector3D>

class TestAlienCellFunctionCommunicator;

class AlienCellFunctionCommunicator : public AlienCellFunction
{
public:
    AlienCellFunctionCommunicator (AlienCell* cell, AlienGrid*& grid);
    AlienCellFunctionCommunicator (AlienCell* cell, quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionCommunicator (AlienCell* cell, QDataStream& stream, AlienGrid*& grid);

    ProcessingResult process (AlienToken* token, AlienCell* previousCell) ;
    CellFunctionType getType () const { return CellFunctionType::COMMUNICATOR; }

    void serialize (QDataStream& stream);
    void getInternalData (quint8* data);

    friend TestAlienCellFunctionCommunicator;

private:
    struct MessageData {
        quint8 channel = 0;
        quint8 message = 0;
        quint8 angle = 0;
        quint8 distance = 0;
    };


    COMMUNICATOR_IN readCommandFromToken (AlienToken* token) const;
    void setListeningChannel (AlienToken* token);

    void sendMessageToNearbyCommunicatorsAndUpdateToken (AlienToken* token, AlienCell* cell, AlienCell* previousCell) const;
    int sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend, AlienCell* senderCell, AlienCell* senderPreviousCell) const;
    QList< AlienCell* > findNearbyCommunicator (AlienCell* cell) const;
    bool sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend, AlienCell* senderCell, AlienCell* senderPreviousCell, AlienCell* receiverCell) const;
    QVector3D calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend, AlienCell* senderCell, AlienCell* senderPreviousCell) const;

    void receiveMessage (AlienToken* token,AlienCell* receiverCell, AlienCell* receiverPreviousCell);
    void calcReceivedMessageAngle (AlienCell* receiverCell, AlienCell* receiverPreviousCell);

    bool _newMessageReceived = false;
    MessageData _receivedMessage;
};

#endif // ALIENCELLFUNCTIONCOMMUNICATOR_H
