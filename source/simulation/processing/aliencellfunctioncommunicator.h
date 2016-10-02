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
    QString getCellFunctionName ();

    //constants for cell function programming
    enum class COMMUNICATOR {
        IN = 26,
        IN_CHANNEL = 27,
        IN_MESSAGE = 28,
        OUT_NUM_MESSAGE_SENT = 29,
        OUT_RECEIVED_BYTE = 30,
        OUT_SENDER_DIRECTION = 31,
        OUT_SENDER_DISTANCE = 32,
    };
    enum class COMMUNICATOR_IN {
        DO_NOTHING,
        SET_LISTENING_CHANNEL,
        SEND_MESSAGE,
        RECEIVE_MESSAGE
    };

private:
    quint8 _listeningChannel;
    quint8 _receivedByte;
    quint8 _senderDirection;
    quint8 _senderDistance;

    void setListeningChannelFromToken (AlienToken* token);
    int sendMessageToNearbyCellsAndReturnNumber (const quint8& channel, const quint8& msg, AlienCell* cell, AlienGrid* grid) const;
    bool sendMessageToCellAndReturnSuccess (const quint8& channel, const quint8& msg, AlienCell* cell) const;
    void receiveMessage () const;

};

#endif // ALIENCELLFUNCTIONCOMMUNICATOR_H
