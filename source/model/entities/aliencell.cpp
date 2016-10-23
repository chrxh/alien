#include "aliencell.h"
#include "aliencellcluster.h"
#include "../processing/aliencellfunction.h"
#include "../processing/aliencellfunctionfactory.h"
#include "../physics/physics.h"
#include "global/globalfunctions.h"
#include "model/simulationsettings.h"

#include <QtCore/qmath.h>


AlienCell* AlienCell::buildCellWithRandomData (qreal energy, AlienGrid*& grid)
{
    return new AlienCell(energy, grid, true);
}

AlienCell* AlienCell::buildCell (qreal energy,
                      AlienGrid*& grid,
                      int maxConnections,
                      int tokenAccessNumber,
                      AlienCellFunction* cellFunction,
                      QVector3D relPos)
{
    return new AlienCell(energy, grid, false, maxConnections, tokenAccessNumber, cellFunction, relPos);
}

AlienCell* AlienCell::buildCell (QDataStream& stream,
                      QMap< quint64, QList< quint64 > >& connectingCells,
                      AlienGrid*& grid)
{
    return new AlienCell(stream, connectingCells, grid);
}

AlienCell* AlienCell::buildCellWithoutConnectingCells (QDataStream& stream,
                      AlienGrid*& grid)
{
    return new AlienCell(stream, grid);
}

AlienCell::~AlienCell()
{
    for(int i = 0; i < _tokenStackPointer; ++i )
        delete _tokenStack[i];
    delete _cellFunction;
    for(int i = 0; i < _newTokenStackPointer; ++i )
        delete _newTokenStack[i];
    if( _maxConnections > 0 )
        delete _connectingCells;
}

bool AlienCell::connectable (AlienCell* otherCell)
{
    return (_numConnections < _maxConnections) && (otherCell->_numConnections < otherCell->_maxConnections);
}

bool AlienCell::isConnectedTo (AlienCell* otherCell)
{
    for( int i = 0; i < _numConnections; ++i )
        if( _connectingCells[i] == otherCell )
            return true;
    return false;
}

void AlienCell::resetConnections (int maxConnections)
{
    //delete old array
    if( _connectingCells )
        delete _connectingCells;

    //set up new array
    _maxConnections = maxConnections;
    _numConnections = 0;
    _connectingCells = new AlienCell*[maxConnections];
}

void AlienCell::newConnection (AlienCell* otherCell)
{
    _connectingCells[_numConnections] = otherCell;
    _numConnections++;
    otherCell->_connectingCells[otherCell->_numConnections] = this;
    otherCell->_numConnections++;
}

void AlienCell::delConnection (AlienCell* otherCell)
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
    for( int i = 0; i < otherCell->_numConnections; ++i ) {
        if( otherCell->_connectingCells[i] == this ) {
            for( int j = i+1; j < otherCell->_numConnections; ++j ) {
                otherCell->_connectingCells[j-1] = otherCell->_connectingCells[j];
            }
            otherCell->_numConnections--;
            break;
        }
    }
}

void AlienCell::delAllConnection ()
{
    for( int i = 0; i < _numConnections; ++i ) {
        AlienCell* otherCell(_connectingCells[i]);
        for( int j = 0; j < otherCell->_numConnections; ++j ) {
            if( otherCell->_connectingCells[j] == this ) {
                for( int k = j+1; k < otherCell->_numConnections; ++k ) {
                    otherCell->_connectingCells[k-1] = otherCell->_connectingCells[k];
                }
                otherCell->_numConnections--;
                break;
            }
        }
    }
    _numConnections = 0;
}

int AlienCell::getNumConnections ()
{
    return _numConnections;
}

int AlienCell::getMaxConnections ()
{
    return _maxConnections;
}

void AlienCell::setMaxConnections (int maxConnections)
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


AlienCell* AlienCell::getConnection (int i)
{
    return _connectingCells[i];
}

QVector3D AlienCell::calcNormal (QVector3D outerSpace, QMatrix4x4& transform)
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
        QVector3D u = (transform.map(_connectingCells[i]->_relPos)-transform.map(_relPos)).normalized();
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
        return transform.map(_relPos)-transform.map(minCell->_relPos);
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

void AlienCell::activatingNewTokens ()
{
    _tokenStackPointer = _newTokenStackPointer;
    for( int i = 0; i < _newTokenStackPointer; ++i ) {
        _tokenStack[i] = _newTokenStack[i];
//        _tokenStack[i]->setTokenAccessNumber(_tokenAccessNumber);
    }
    _newTokenStackPointer = 0;
}

const quint64& AlienCell::getId ()
{
    return _id;
}

void AlienCell::setId (quint64 id)
{
    _id = id;
}

const quint64& AlienCell::getTag ()
{
    return _tag;
}

void AlienCell::setTag (quint64 tag)
{
    _tag = tag;
}

int AlienCell::getNumToken (bool newTokenStackPointer)
{
    if( newTokenStackPointer )
        return _newTokenStackPointer;
    else
        return _tokenStackPointer;
}

AlienToken* AlienCell::getToken (int i)
{
    return _tokenStack[i];
}

void AlienCell::addToken (AlienToken* token, bool activateNow, bool setAccessNumber)
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

void AlienCell::delAllTokens ()
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

void AlienCell::setCluster (AlienCellCluster* cluster)
{
    _cluster = cluster;
}

AlienCellCluster* AlienCell::getCluster()
{
    return _cluster;
}

QVector3D AlienCell::calcPosition (bool topologyCorrection)
{
    return _cluster->calcPosition(this, topologyCorrection);
}

void AlienCell::setAbsPosition (QVector3D pos)
{
    _relPos = _cluster->absToRelPos(pos);
//    QMatrix4x4 clusterTransform(_cluster->_transform.inverted());
//    _relPos = clusterTransform.map(pos);
}

void AlienCell::setAbsPositionAndUpdateMap (QVector3D pos)
{
    QVector3D oldPos(calcPosition());
    if( _grid->getCell(oldPos) == this )
        _grid->setCell(oldPos, 0);
    _relPos = _cluster->absToRelPos(pos);
    if( _grid->getCell(pos) == 0 )
        _grid->setCell(pos, this);
}

QVector3D AlienCell::getRelPos ()
{
    return _relPos;
}

void AlienCell::setRelPos (QVector3D relPos)
{
    _relPos = relPos;
}


AlienCellFunction* AlienCell::getCellFunction ()
{
    return _cellFunction;
}

void AlienCell::setCellFunction (AlienCellFunction* cellFunction)
{
    if( _cellFunction )
        delete _cellFunction;
    _cellFunction = cellFunction;
}

int AlienCell::getTokenAccessNumber ()
{
    return _tokenAccessNumber;
}

void AlienCell::setTokenAccessNumber (int i)
{
    _tokenAccessNumber = i % simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
}

bool AlienCell::blockToken ()
{
    return _blockToken;
}

void AlienCell::setBlockToken (bool block)
{
    _blockToken = block;
}

qreal AlienCell::getEnergy()
{
    return _energy;
}

qreal AlienCell::getEnergyIncludingTokens()
{
    qreal energy = _energy;
    for(int i = 0; i < _tokenStackPointer; ++i)
        energy += _tokenStack[i]->energy;
    for(int i = 0; i < _newTokenStackPointer; ++i)
        energy += _newTokenStack[i]->energy;
    return energy;
}

void AlienCell::setEnergy (qreal i)
{
    _energy = i;
}

QVector< quint8 >& AlienCell::getMemory ()
{
    return _memory;
}

void AlienCell::serialize (QDataStream& stream)
{
    //cell function
    _cellFunction->serialize(stream);

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
        stream << _connectingCells[i]->_id;
    }

    //remaining data
    stream << _tokenAccessNumber << _blockToken << _vel << _color;
    stream << simulationParameters.CELL_MEMSIZE;
    for(int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
        stream << _memory[i];
}

QVector3D AlienCell::getVel ()
{
    return _vel;
}

void AlienCell::setVel (QVector3D vel)
{
    _vel = vel;
}

quint8 AlienCell::getColor ()
{
    return _color;
}

void AlienCell::setColor (quint8 color)
{
    _color = color;
}

AlienCell::AlienCell (qreal energy,
                      AlienGrid*& grid,
                      bool random,
                      int maxConnections,
                      int tokenAccessNumber,
                      AlienCellFunction* cellFunction,
                      QVector3D relPos)
    : _grid(grid),
      _cellFunction(cellFunction),
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
        _cellFunction = AlienCellFunctionFactory::buildRandom(random, _grid);
        for( int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
            _memory[i] = qrand()%256;
    }
    else {
        resetConnections(maxConnections);
        if( !cellFunction )
            _cellFunction = AlienCellFunctionFactory::build("COMPUTER", false, _grid);     //standard cell function
        for( int i = 0; i < simulationParameters.CELL_MEMSIZE; ++i )
            _memory[i] = 0;
    }
}

AlienCell::AlienCell (QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, AlienGrid*& grid)
    : _grid(grid),
      _tokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _newTokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _memory(simulationParameters.CELL_MEMSIZE)
{

    //cell function
    _cellFunction = AlienCellFunctionFactory::build(stream, _grid);

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

AlienCell::AlienCell (QDataStream& stream, AlienGrid*& grid)
    : _grid(grid),
      _tokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _newTokenStack(simulationParameters.CELL_TOKENSTACKSIZE),
      _memory(simulationParameters.CELL_MEMSIZE)
{

    //cell function
    _cellFunction = AlienCellFunctionFactory::build(stream, _grid);

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



