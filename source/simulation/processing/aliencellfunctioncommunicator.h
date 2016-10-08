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
        OUT_NUM_MESSAGE_SENT = 29,
        OUT_RECEIVED_MESSAGE = 30,
        OUT_RECEIVED_ANGLE = 31,
        OUT_RECEIVED_DISTANCE = 32,
    };
    enum class COMMUNICATOR_IN {
        DO_NOTHING = 0,
        SET_LISTENING_CHANNEL = 1,
        SEND_MESSAGE = 2,
        RECEIVE_MESSAGE = 3
    };

private:
    quint8 _listeningChannel;
    quint8 _receivedMsg;
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
