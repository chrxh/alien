#ifndef ALIENCELLFUNCTIONCOMMUNICATOR_H
#define ALIENCELLFUNCTIONCOMMUNICATOR_H

#include "aliencellfunction.h"

#include <QVector3D>

class AlienCellFunctionCommunicator : public AlienCellFunction
{
public:
    AlienCellFunctionCommunicator (AlienGrid*& grid);
    AlienCellFunctionCommunicator (quint8* cellTypeData, AlienGrid*& grid);
    AlienCellFunctionCommunicator (QDataStream& stream, AlienGrid*& grid);

    void execute (AlienToken* token,
                  AlienCell* cell,
                  AlienCell* previousCell,
                  AlienEnergy*& newParticle,
                  bool& decompose);
    QString getCellFunctionName () const;
    void serialize (QDataStream& stream);


    //constants for cell function programming
    enum class COMMUNICATOR {
        IN = 26,
        IN_CHANNEL = 27,
        IN_MESSAGE = 28,
        IN_ANGLE = 29,
        IN_DISTANCE = 30,
        OUT_SENT_NUM_MESSAGE = 31,
        OUT_RECEIVED_NEW_MESSAGE = 32,
        OUT_RECEIVED_MESSAGE = 33,
        OUT_RECEIVED_ANGLE = 34,
        OUT_RECEIVED_DISTANCE = 35,
    };
    enum class COMMUNICATOR_IN {
        DO_NOTHING,
        SET_LISTENING_CHANNEL,
        SEND_MESSAGE,
        RECEIVE_MESSAGE
    };
    enum class COMMUNICATOR_OUT_RECEIVED_NEW_MESSAGE {
        NO_NEW_MESSAGE,
        NEW_MESSAGE
    };

private:
    bool _newMessageReceived;
    struct MessageData {
        quint8 channel = 0;
        quint8 message = 0;
        quint8 angle = 0;
        quint8 distance = 0;
    } _receivedMessage;


    COMMUNICATOR_IN readCommandFromToken (AlienToken* token) const;
    void setListeningChannel (AlienToken* token);

    void sendMessageToNearbyCommunicatorsAndUpdateToken (AlienToken* token, AlienCell* cell, AlienCell* previousCell) const;
    int sendMessageToNearbyCommunicatorsAndReturnNumber (const MessageData& messageToSend, AlienCell* senderCell, AlienCell* senderPreviousCell) const;
    QList< AlienCell* > findNearbyCommunicator (AlienCell* cell) const;
    bool sendMessageToCommunicatorAndReturnSuccess (const MessageData& messageToSend, AlienCell* senderCell, AlienCell* senderPreviousCell, AlienCell* receiverCell) const;
    AlienCellFunctionCommunicator* getCommunicator (AlienCell* cell) const;
    QVector3D calcDisplacementOfObjectFromSender (const MessageData& messageToSend, AlienCell* senderCell, AlienCell* senderPreviousCell) const;

    void receiveMessage (AlienToken* token,AlienCell* receiverCell, AlienCell* receiverPreviousCell);
    void calcReceivedMessageAngle (AlienCell* receiverCell, AlienCell* receiverPreviousCell);
};

#endif // ALIENCELLFUNCTIONCOMMUNICATOR_H
