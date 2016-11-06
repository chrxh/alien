#include "aliencellimpl.h"

#include "model/entities/aliencellcluster.h"
#include "model/entities/alientoken.h"
#include "model/physics/physics.h"
#include "model/simulationsettings.h"

#include "global/global.h"

#include <QtCore/qmath.h>

AlienCellImpl::AlienCellImpl (qreal energy, AlienGrid*& grid, bool random,
                              int maxConnections, int tokenAccessNumber, QVector3D relPos)
    : AlienCell(grid),
      _tokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _newTokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _tokenStackPointer(0),
      _newTokenStackPointer(0),
      _toBeKilled(false),
      _tag(0),
      _id(GlobalFunctions::getTag()),
      _protectionCounter(0),
      _relPos(relPos),
      _cluster(0),
      _energy(energy),
      _maxConnections(0),
      _numConnections(0),
      _connectingCells(0),
      _tokenAccessNumber(tokenAccessNumber),
      _blockToken(false),
      _memory(simulationParameters.CELL_MEMSIZE),
      _vel(0.0, 0.0, 0.0),
      _color(0)
{
    //set initial values
    if( random ) {
        resetConnections(qrand() % (simulationParameters.MAX_CELL_CONNECTIONS+1));
        _tokenAccessNumber = qrand() % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
        for( int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
            _memory[i] = qrand()%256;
    }
    else {
        resetConnections(maxConnections);
        for( int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
            _memory[i] = 0;
    }
}

AlienCellImpl::AlienCellImpl (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells,
                              AlienGrid*& grid)
    : AlienCell(grid),
      _tokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _newTokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _memory(simulationParameters.CELL_MEMSIZE)
{

    //token stack
    stream >> _tokenStackPointer;
    for( int i = 0; i < _tokenStackPointer; ++i ) {
        if( i < simulationParameters.CELL_TOKENSTACKSIZE )
            _tokenStack[i] = new AlienToken(stream);
        else{
            //dummy token
            AlienToken* temp = new AlienToken(stream);
            delete temp;
        }
    }
    if( _tokenStackPointer > simulationParameters.CELL_TOKENSTACKSIZE )
        _tokenStackPointer = simulationParameters.CELL_TOKENSTACKSIZE;
    _newTokenStackPointer = 0;

    //remaining data
    int numConnections(0);
    stream >> _toBeKilled >> _tag >> _id >> _protectionCounter >> _relPos
           >> _energy >> _maxConnections >> numConnections;

    //connecting cells
    _connectingCells = 0;
    resetConnections(_maxConnections);
    _numConnections = numConnections;
    for( int i = 0; i < _numConnections; ++i) {
        quint64 id;
        stream >> id;
        connectingCells[_id] << id;
    }

    //remaining data
    stream >> _tokenAccessNumber >> _blockToken >> _vel >> _color;

    //cell memory
    int memSize;
    stream >> memSize;
    quint8 data;
    for(int i = 0; i < memSize; ++i ) {
        if( i < simulationParameters.CELL_MEMSIZE ) {
            stream >> data;
            _memory[i] = data;
        }
        else {
            stream >> data;
        }
    }
    for(int i = memSize; i < simulationParameters.CELL_MEMSIZE; ++i)
        _memory[i] = 0;
}

AlienCellImpl::AlienCellImpl (QDataStream& stream, AlienGrid*& grid)
    : AlienCell(grid),
      _tokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _newTokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _memory(simulationParameters.CELL_MEMSIZE)
{

    //token stack
    stream >> _tokenStackPointer;
    for( int i = 0; i < _tokenStackPointer; ++i ) {
        if( i < simulationParameters.CELL_TOKENSTACKSIZE )
            _tokenStack[i] = new AlienToken(stream);
        else{
            //dummy token
            AlienToken* temp = new AlienToken(stream);
            delete temp;
        }
    }
    if( _tokenStackPointer > simulationParameters.CELL_TOKENSTACKSIZE )
        _tokenStackPointer = simulationParameters.CELL_TOKENSTACKSIZE;
    _newTokenStackPointer = 0;

    //remaining data
    _numConnections = 0;
    int numConnections = 0;
    stream >> _toBeKilled >> _tag >> _id >> _protectionCounter >> _relPos
           >> _energy >> _maxConnections >> numConnections;

    //connecting cells (NOTE: here they are just read but not established)
    _connectingCells = 0;
    resetConnections(_maxConnections);
    for( int i = 0; i < numConnections; ++i) {
        quint64 id;
        stream >> id;
    }

    //remaining data
    stream >> _tokenAccessNumber >> _blockToken >> _vel >> _color;

    //cell memory
    int memSize;
    stream >> memSize;
    quint8 data;
    for(int i = 0; i < memSize; ++i ) {
        if( i < simulationParameters.CELL_MEMSIZE ) {
            stream >> data;
            _memory[i] = data;
        }
        else {
            stream >> data;
        }
    }
    for(int i = memSize; i < simulationParameters.CELL_MEMSIZE; ++i)
        _memory[i] = 0;
}

AlienCellImpl::~AlienCellImpl()
{
    for(int i = 0; i < _tokenStackPointer; ++i )
        delete _tokenStack[i];
    for(int i = 0; i < _newTokenStackPointer; ++i )
        delete _newTokenStack[i];
    if( _maxConnections > 0 )
        delete _connectingCells;
}

AlienCell::ProcessingResult AlienCellImpl::process (AlienToken* token, AlienCell* previousCell)
{
    return {false, 0};
}

bool AlienCellImpl::connectable (AlienCell* otherCell) const
{
    return (_numConnections < _maxConnections) && (otherCell->getNumConnections() < otherCell->getMaxConnections());
}

bool AlienCellImpl::isConnectedTo (AlienCell* otherCell) const
{
    for( int i = 0; i < _numConnections; ++i )
        if( _connectingCells[i] == otherCell )
            return true;
    return false;
}

void AlienCellImpl::resetConnections (int maxConnections)
{
    //delete old array
    if( _connectingCells )
        delete _connectingCells;

    //set up new array
    _maxConnections = maxConnections;
    _numConnections = 0;
    _connectingCells = new AlienCell*[maxConnections];
}

void AlienCellImpl::newConnection (AlienCell* otherCell)
{
    _connectingCells[_numConnections] = otherCell;
    _numConnections++;
    otherCell->setConnection(otherCell->getNumConnections(), this);
    otherCell->setNumConnections(otherCell->getNumConnections()+1);
}

void AlienCellImpl::delConnection (AlienCell* otherCell)
{
    for( int i = 0; i < _numConnections; ++i ) {
        if( _connectingCells[i] == otherCell ) {
            for( int j = i+1; j < _numConnections; ++j ) {
                _connectingCells[j-1] = _connectingCells[j];
            }
            _numConnections--;
            break;
        }
    }
    for( int i = 0; i < otherCell->getNumConnections(); ++i ) {
        if( otherCell->getConnection(i) == this ) {
            for( int j = i+1; j < otherCell->getNumConnections(); ++j ) {
                otherCell->setConnection(j-1, otherCell->getConnection(j));
            }
            otherCell->setNumConnections(otherCell->getNumConnections()-1);
            break;
        }
    }
}

void AlienCellImpl::delAllConnection ()
{
    for( int i = 0; i < _numConnections; ++i ) {
        AlienCell* otherCell(_connectingCells[i]);
        for( int j = 0; j < otherCell->getNumConnections(); ++j ) {
            if( otherCell->getConnection(j) == this ) {
                for( int k = j+1; k < otherCell->getNumConnections(); ++k ) {
                    otherCell->setConnection(k-1, otherCell->getConnection(k));
                }
                otherCell->setNumConnections(otherCell->getNumConnections()-1);
                break;
            }
        }
    }
    _numConnections = 0;
}

int AlienCellImpl::getNumConnections () const
{
    return _numConnections;
}

void AlienCellImpl::setNumConnections (int num)
{
    _numConnections = num;
}

int AlienCellImpl::getMaxConnections () const
{
    return _maxConnections;
}

void AlienCellImpl::setMaxConnections (int maxConnections)
{

    //new array
    AlienCell** newArray = new AlienCell*[maxConnections];
    if( _connectingCells ) {

        //copy old array
        for(int i = 0 ; i < qMin(_maxConnections, maxConnections); ++i) {
            newArray[i] = _connectingCells[i];
        }

        //delete old array
        delete _connectingCells;
    }
    _maxConnections = maxConnections;
    if( _numConnections > _maxConnections )
        _numConnections = _maxConnections;
    _connectingCells = newArray;
}


AlienCell* AlienCellImpl::getConnection (int i) const
{
    return _connectingCells[i];
}

void AlienCellImpl::setConnection (int i, AlienCell* cell)
{
    _connectingCells[i] = cell;
}

QVector3D AlienCellImpl::calcNormal (QVector3D outerSpace, QMatrix4x4& transform) const
{
    if( _numConnections < 2 ) {
        return outerSpace.normalized();
    }

    //find adjacent cells to the outerSpace vector
    outerSpace.normalize();
    AlienCell* minCell(0);
    AlienCell* maxCell(0);
    QVector3D minVector(0.0, 0.0, 0.0);
    QVector3D maxVector(0.0, 0.0, 0.0);
    qreal minH(0.0);
    qreal maxH(0.0);

    for(int i = 0; i < _numConnections; ++i) {

        //calculate h (angular distance from outerSpace vector)
        QVector3D u = (transform.map(_connectingCells[i]->getRelPos())-transform.map(_relPos)).normalized();
        qreal h = QVector3D::dotProduct(outerSpace, u);
        if( (outerSpace.x()*u.y()-outerSpace.y()*u.x()) < 0.0 )
            h = -2 - h;

        //save min and max
        if( (!minCell) || (h < minH) ) {
            minCell = _connectingCells[i];
            minVector = u;
            minH = h;
        }
        if( (!maxCell) || (h > maxH) ) {
            maxCell = _connectingCells[i];
            maxVector = u;
            maxH = h;
        }
    }

    //no adjacent cells?
    if( (!minCell) && (!maxCell) ) {
        return outerSpace;
    }

    //one adjacent cells?
    if( minCell == maxCell ) {
        return transform.map(_relPos)-transform.map(minCell->getRelPos());
    }

    //calc normal vectors
    qreal temp = maxVector.x();
    maxVector.setX(maxVector.y());
    maxVector.setY(-temp);
    temp = minVector.x();
    minVector.setX(-minVector.y());
    minVector.setY(temp);
    return minVector+maxVector;
}

void AlienCellImpl::activatingNewTokens ()
{
    _tokenStackPointer = _newTokenStackPointer;
    for( int i = 0; i < _newTokenStackPointer; ++i ) {
        _tokenStack[i] = _newTokenStack[i];
//        _tokenStack[i]->setTokenAccessNumber(_tokenAccessNumber);
    }
    _newTokenStackPointer = 0;
}

const quint64& AlienCellImpl::getId () const
{
    return _id;
}

void AlienCellImpl::setId (quint64 id)
{
    _id = id;
}

const quint64& AlienCellImpl::getTag () const
{
    return _tag;
}

void AlienCellImpl::setTag (quint64 tag)
{
    _tag = tag;
}

int AlienCellImpl::getNumToken (bool newTokenStackPointer) const
{
    if( newTokenStackPointer )
        return _newTokenStackPointer;
    else
        return _tokenStackPointer;
}

AlienToken* AlienCellImpl::getToken (int i) const
{
    return _tokenStack[i];
}

void AlienCellImpl::addToken (AlienToken* token, bool activateNow, bool setAccessNumber)
{
    if( setAccessNumber )
        token->setTokenAccessNumber(_tokenAccessNumber);
    if( activateNow ) {
        _tokenStack[_tokenStackPointer] = token;
        ++_tokenStackPointer;
    }
    else {
        _newTokenStack[_newTokenStackPointer] = token;
        ++_newTokenStackPointer;
    }
}

void AlienCellImpl::delAllTokens ()
{
/*    for( int j = i+1; j < _tokenStackPointer; ++j )
        _tokenStack[j-1] = _tokenStack[j];
    --_tokenStackPointer;*/
    for( int j = 0; j < _tokenStackPointer; ++j )
         delete _tokenStack[j];
    for( int j = 0; j < _newTokenStackPointer; ++j )
         delete _newTokenStack[j];
    _tokenStackPointer = 0;
    _newTokenStackPointer = 0;
}

void AlienCellImpl::setCluster (AlienCellCluster* cluster)
{
    _cluster = cluster;
}

AlienCellCluster* AlienCellImpl::getCluster() const
{
    return _cluster;
}

QVector3D AlienCellImpl::calcPosition (bool topologyCorrection) const
{
    return _cluster->calcPosition(this, topologyCorrection);
}

void AlienCellImpl::setAbsPosition (QVector3D pos)
{
    _relPos = _cluster->absToRelPos(pos);
}

void AlienCellImpl::setAbsPositionAndUpdateMap (QVector3D pos)
{
    QVector3D oldPos(calcPosition());
    if( _grid->getCell(oldPos) == this )
        _grid->setCell(oldPos, 0);
    _relPos = _cluster->absToRelPos(pos);
    if( _grid->getCell(pos) == 0 )
        _grid->setCell(pos, this);
}

QVector3D AlienCellImpl::getRelPos () const
{
    return _relPos;
}

void AlienCellImpl::setRelPos (QVector3D relPos)
{
    _relPos = relPos;
}


int AlienCellImpl::getTokenAccessNumber () const
{
    return _tokenAccessNumber;
}

void AlienCellImpl::setTokenAccessNumber (int i)
{
    _tokenAccessNumber = i % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
}

bool AlienCellImpl::isTokenBlocked () const
{
    return _blockToken;
}

void AlienCellImpl::setTokenBlocked (bool block)
{
    _blockToken = block;
}

qreal AlienCellImpl::getEnergy() const
{
    return _energy;
}

qreal AlienCellImpl::getEnergyIncludingTokens() const
{
    qreal energy = _energy;
    for(int i = 0; i < _tokenStackPointer; ++i)
        energy += _tokenStack[i]->energy;
    for(int i = 0; i < _newTokenStackPointer; ++i)
        energy += _newTokenStack[i]->energy;
    return energy;
}

void AlienCellImpl::setEnergy (qreal i)
{
    _energy = i;
}

QVector< quint8 >& AlienCellImpl::getMemoryReference ()
{
    return _memory;
}

void AlienCellImpl::serialize (QDataStream& stream) const
{
    //token
    stream << _tokenStackPointer;
    for( int i = 0; i < _tokenStackPointer; ++i) {
        _tokenStack[i]->serialize(stream);
    }

    //remaining data
    stream << _toBeKilled << _tag << _id << _protectionCounter << _relPos
           << _energy << _maxConnections << _numConnections;

    //connecting cells
    for( int i = 0; i < _numConnections; ++i) {
        stream << _connectingCells[i]->getId();
    }

    //remaining data
    stream << _tokenAccessNumber << _blockToken << _vel << _color;
    stream << simulationParameters.CELL_MEMSIZE;
    for(int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
        stream << _memory[i];
}

QVector3D AlienCellImpl::getVel () const
{
    return _vel;
}

void AlienCellImpl::setVel (QVector3D vel)
{
    _vel = vel;
}

quint8 AlienCellImpl::getColor () const
{
    return _color;
}

void AlienCellImpl::setColor (quint8 color)
{
    _color = color;
}
