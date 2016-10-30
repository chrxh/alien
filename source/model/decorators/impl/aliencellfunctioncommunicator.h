#ifndef ALIENCELLFUNCTIONCOMMUNICATOR_H
#define ALIENCELLFUNCTIONCOMMUNICATOR_H

#include "model/decorators/aliencellfunction.h"

#include <QVector3D>

class TestAlienCellFunctionCommunicator;

class AlienCellFunctionCommunicator : public AlienCellFunction
{
public:
    AlienCellFunctionCommunicator (AlienGrid*& grid);
    AlienCellFunctionCommunicator (quint8* cellFunctionData, AlienGrid*& grid);
    AlienCellFunctionCommunicator (QDataStream& stream, AlienGrid*& grid);

    virtual ProcessingResult process (AlienToken* token, AlienCell* previousCell) = 0;
    void process (AlienToken* token,
                  AlienCell* cell,
                  AlienCell* previousCell,
                  AlienEnergy*& newParticle,
                  bool& decompose);
    void serialize (QDataStream& stream);
    void getInternalData (quint8* data);

    friend TestAlienCellFunctionCommunicator;

private:
    bool _newMessageReceived = false;
    struct MessageData {
        quint8 channel = 0;
        quint8 message = 0;
        quint8 angle = 0;
        quint8 distance = 0;
    } _receivedMessage;

    COMMUNICATOR_IN readCommandFromToken (AlienToken* token) const;
    void setListeningChannel (AlienToken* token);

    void sendMessageToNearbyCommunicatorsAndUpdateToken (AlienToken* token, AlienCell* cell, AlienCell* previousCell) const;
    int sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageDataToSend, AlienCell* senderCell, AlienCell* senderPreviousCell) const;
    QList< AlienCell* > findNearbyCommunicator (AlienCell* cell) const;
    bool sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageDataToSend, AlienCell* senderCell, AlienCell* senderPreviousCell, AlienCell* receiverCell) const;
    AlienCellFunctionCommunicator* getCommunicator (AlienCell* cell) const;
    QVector3D calcDisplacementOfObjectFromSender (const MessageData& messageDataToSend, AlienCell* senderCell, AlienCell* senderPreviousCell) const;

    void receiveMessage (AlienToken* token,AlienCell* receiverCell, AlienCell* receiverPreviousCell);
    void calcReceivedMessageAngle (AlienCell* receiverCell, AlienCell* receiverPreviousCell);
};

#endif // ALIENCELLFUNCTIONCOMMUNICATOR_H
