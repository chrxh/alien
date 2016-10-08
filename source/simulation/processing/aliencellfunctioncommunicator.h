#ifndef ALIENCELLFUNCTIONCOMMUNICATOR_H
#define ALIENCELLFUNCTIONCOMMUNICATOR_H

#include "aliencellfunction.h"

class AlienCellFunctionCommunicator : public AlienCellFunction
{
public:
    AlienCellFunctionCommunicator ();
    AlienCellFunctionCommunicator (quint8* cellTypeData);
    AlienCellFunctionCommunicator (QDataStream& stream);

    void execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose);
    QString getCellFunctionName () const;

    //constants for cell function programming
    enum class COMMUNICATOR {
        IN = 26,
        IN_CHANNEL = 27,
        IN_MESSAGE = 28,
        OUT_SENT_NUM_MESSAGE = 29,
        OUT_RECEIVED_NEW_MESSAGE = 30,
        OUT_RECEIVED_MESSAGE = 31,
        OUT_RECEIVED_ANGLE = 32,
        OUT_RECEIVED_DISTANCE = 33,
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
    quint8 _listeningChannel;
    bool _receivedNewMessage;
    quint8 _receivedMessage;
    quint8 _receivedAngle;
    quint8 _receivedDistance;

    COMMUNICATOR_IN readCommandFromToken (AlienToken* token) const;
    void sendMessageToNearbyCellsAndUpdateToken (AlienToken* token, AlienCell* cell, AlienGrid* grid) const;
    quint8 readListeningChannelFrom (AlienToken* token) const;
    int sendMessageToNearbyCellsAndReturnNumber (const quint8& channel, const quint8& msg, AlienCell* cell, AlienGrid* grid) const;
    QList< AlienCell* > findNearbyCommunicatorCells (AlienCell* cell, AlienGrid* grid) const;
    bool sendMessageToCellAndReturnSuccess (const quint8& channel, const quint8& msg, AlienCell* senderCell, AlienCell* receiverCell, AlienGrid* grid) const;
    AlienCellFunctionCommunicator* getCommunicator (AlienCell* cell) const;
    void receiveMessage () const;

};

#endif // ALIENCELLFUNCTIONCOMMUNICATOR_H
