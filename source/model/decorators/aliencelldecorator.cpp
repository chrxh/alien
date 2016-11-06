#include "aliencelldecorator.h"

AlienCellDecorator::~AlienCellDecorator ()
{
    delete _cell;
}

void AlienCellDecorator::serialize (QDataStream& stream) const
{
    _cell->serialize(stream);
}


AlienCell::ProcessingResult AlienCellDecorator::process (AlienToken* token, AlienCell* previousCell)
{
    return _cell->process(token, previousCell);
}

bool AlienCellDecorator::connectable (AlienCell* otherCell) const
{
    return _cell->connectable(otherCell);
}

bool AlienCellDecorator::isConnectedTo (AlienCell* otherCell) const
{
    return _cell->isConnectedTo(otherCell);
}

void AlienCellDecorator::resetConnections (int maxConnections)
{
    _cell->resetConnections(maxConnections);
}

void AlienCellDecorator::newConnection (AlienCell* otherCell)
{
    _cell->newConnection(otherCell);
}

void AlienCellDecorator::delConnection (AlienCell* otherCell)
{
    _cell->delConnection(otherCell);
}

void AlienCellDecorator::delAllConnection ()
{
    _cell->delAllConnection();
}

int AlienCellDecorator::getNumConnections () const
{
    return _cell->getNumConnections();
}

int AlienCellDecorator::getMaxConnections () const
{
    return _cell->getMaxConnections();
}

void AlienCellDecorator::setMaxConnections (int maxConnections)
{
    _cell->setMaxConnections(maxConnections);
}

AlienCell* AlienCellDecorator::getConnection (int i) const
{
    return _cell->getConnection(i);
}

void AlienCellDecorator::setConnection (int i, AlienCell* cell)
{
    _cell->setConnection(i, cell);
}

QVector3D AlienCellDecorator::calcNormal (QVector3D outerSpace, QMatrix4x4& transform) const
{
    return _cell->calcNormal(outerSpace, transform);
}

void AlienCellDecorator::activatingNewTokens ()
{
    _cell->activatingNewTokens();
}

const quint64& AlienCellDecorator::getId () const
{
    return _cell->getId();
}

void AlienCellDecorator::setId (quint64 id)
{
    _cell->setId(id);
}

const quint64& AlienCellDecorator::getTag () const
{
    return _cell->getTag();
}

void AlienCellDecorator::setTag (quint64 tag)
{
    _cell->setTag(tag);
}

int AlienCellDecorator::getNumToken (bool newTokenStackPointer) const
{
    return _cell->getNumToken(newTokenStackPointer);
}

AlienToken* AlienCellDecorator::getToken (int i) const
{
    return _cell->getToken(i);
}

void AlienCellDecorator::addToken (AlienToken* token, bool activateNow, bool updateAccessNumber)
{
    _cell->addToken(token, activateNow, updateAccessNumber);
}

void AlienCellDecorator::delAllTokens ()
{
    _cell->delAllTokens();
}

void AlienCellDecorator::setCluster (AlienCellCluster* cluster)
{
    _cell->setCluster(cluster);
}

AlienCellCluster* AlienCellDecorator::getCluster () const
{
    return _cell->getCluster();
}

QVector3D AlienCellDecorator::calcPosition (bool topologyCorrection) const
{
    return _cell->calcPosition(topologyCorrection);
}

void AlienCellDecorator::setAbsPosition (QVector3D pos)
{
    _cell->setAbsPosition(pos);
}

void AlienCellDecorator::setAbsPositionAndUpdateMap (QVector3D pos)
{
    _cell->setAbsPositionAndUpdateMap(pos);
}

QVector3D AlienCellDecorator::getRelPos () const
{
    return _cell->getRelPos();
}

void AlienCellDecorator::setRelPos (QVector3D relPos)
{
    _cell->setRelPos(relPos);
}

int AlienCellDecorator::getTokenAccessNumber () const
{
    return _cell->getTokenAccessNumber();
}

void AlienCellDecorator::setTokenAccessNumber (int i)
{
    _cell->setTokenAccessNumber(i);
}

bool AlienCellDecorator::isTokenBlocked () const
{
    return _cell->blockToken();
}

void AlienCellDecorator::setTokenBlocked (bool block)
{
    _cell->setBlockToken(block);
}

qreal AlienCellDecorator::getEnergy()
{
    return _cell->getEnergy();
}

qreal AlienCellDecorator::getEnergyIncludingTokens() const
{
    return _cell->getEnergyIncludingTokens();
}

void AlienCellDecorator::setEnergy (qreal i)
{
    _cell->setEnergy(i);
}

QVector< quint8 >& AlienCellDecorator::getMemoryReference ()
{
    return _cell->getMemory();
}

void AlienCellDecorator::serialize (QDataStream& stream)
{
    _cell->serialize(stream);
}

QVector3D AlienCellDecorator::getVel () const
{
    return _cell->getVel();
}

void AlienCellDecorator::setVel (QVector3D vel)
{
    _cell->setVel(vel);
}

quint8 AlienCellDecorator::getColor () const
{
    return _cell->getColor();
}

void AlienCellDecorator::setColor (quint8 color)
{
    _cell->setColor(color);
}
