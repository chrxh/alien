#include "aliencellfunctioncommunicator.h"
#include "../entities/aliencell.h"
#include "../entities/aliencellcluster.h"

#include <QString>

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator()
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (quint8* cellTypeData)
{

}

AlienCellFunctionCommunicator::AlienCellFunctionCommunicator (QDataStream& stream)
{

}

void AlienCellFunctionCommunicator::execute (AlienToken* token, AlienCell* previousCell, AlienCell* cell, AlienGrid* grid, AlienEnergy*& newParticle, bool& decompose)
{
    quint8 cmd = token->memory[static_cast<int>(COMMUNICATOR::IN)] % 4;

    if( cmd == static_cast<int>(COMMUNICATOR_IN::SET_LISTENING_CHANNEL) )
        setListeningChannelFromToken(token);
    if( cmd == static_cast<int>(COMMUNICATOR_IN::SEND_MESSAGE) )
        sendMessageFromTokenToNearbyCells(token, cell, grid);
    if( cmd == static_cast<int>(COMMUNICATOR_IN::RECEIVE_MESSAGE) )
        receiveMessage();
}

QString AlienCellFunctionCommunicator::getCellFunctionName ()
{
    return "COMMUNICATOR";
}

void AlienCellFunctionCommunicator::setListeningChannelFromToken (AlienToken* token)
{
    _listeningChannel = token->memory[static_cast<int>(COMMUNICATOR::IN_CHANNEL)];
}

void AlienCellFunctionCommunicator::sendMessageFromTokenToNearbyCells (AlienToken* token, AlienCell* cell, AlienGrid* grid)
{
    QVector3D cellPos = cell->calcPosition();
    QSet< AlienCellCluster* > nearbyClusters = grid->getNearbyClusters(cellPos, 100.0);
}

void AlienCellFunctionCommunicator::receiveMessage ()
{

}

