#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"
#include "Model/Local/Cluster.h"
#include "Model/Local/Token.h"
#include "Model/Local/CellFeature.h"
#include "Model/Local/Physics.h"
#include "Model/Api/SimulationParameters.h"
#include "Model/Local/UnitContext.h"
#include "Model/Local/CellMap.h"

#include "CellImpl.h"

CellImpl::CellImpl (uint64_t id, qreal energy, UnitContext* context, int maxConnections, int tokenBranchNumber)
	: _id(id)
	, _context(context)
	, _tokenStack(context->getSimulationParameters()->cellMaxToken)
	, _newTokenStack(context->getSimulationParameters()->cellMaxToken)
{
	_energy = energy;
    _tokenBranchNumber = tokenBranchNumber;
    resetConnections(maxConnections);
}

CellImpl::~CellImpl()
{
    for(int i = 0; i < _tokenStackPointer; ++i )
        delete _tokenStack[i];
    for(int i = 0; i < _newTokenStackPointer; ++i )
        delete _newTokenStack[i];
    if( _maxConnections > 0 )
        delete _connectingCells;
    delete _features;
}

void CellImpl::setContext(UnitContext * context)
{
	_context = context;
	for (int i = 0; i < _tokenStackPointer; ++i) {
		_tokenStack[i]->setContext(context);
	}
	if (_features) {
		_features->setContext(context);
	}
}

CellDescription CellImpl::getDescription(ResolveDescription const& resolveDescription) const
{
	CellDescription result;
	result.setId(_id).setPos(calcPosition()).setMaxConnections(_maxConnections)
		.setTokenBranchNumber(_tokenBranchNumber).setEnergy(_energy).setMetadata(_metadata)
		.setFlagTokenBlocked(_blockToken).setCellFeature(_features->getDescription());
	if (resolveDescription.resolveCellLinks) {
		list<uint64_t> connectingCells;
		for (int i = 0; i < _numConnections; ++i) {
			connectingCells.push_back(_connectingCells[i]->getId());
		}
		result.setConnectingCells(connectingCells);
	}
	return result;
}

void CellImpl::registerFeatures (CellFeature* features)
{
    _features = features;
}

CellFeature* CellImpl::getFeatures () const
{
    return _features;
}

void CellImpl::removeFeatures ()
{
    delete _features;
    _features = nullptr;
}

bool CellImpl::connectable (Cell* otherCell) const
{
    return (_numConnections < _maxConnections) && (otherCell->getNumConnections() < otherCell->getMaxConnections());
}

bool CellImpl::isConnectedTo (Cell* otherCell) const
{
    for( int i = 0; i < _numConnections; ++i )
        if( _connectingCells[i] == otherCell )
            return true;
    return false;
}

void CellImpl::resetConnections (int maxConnections)
{
    //delete old array
    delete _connectingCells;

    //set up new array
    _maxConnections = maxConnections;
    _numConnections = 0;
    _connectingCells = new Cell*[maxConnections];
}

void CellImpl::newConnection (Cell* otherCell)
{
    _connectingCells[_numConnections] = otherCell;
    _numConnections++;
    otherCell->setConnection(otherCell->getNumConnections(), this);
    otherCell->setNumConnections(otherCell->getNumConnections()+1);
}

void CellImpl::delConnection (Cell* otherCell)
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

void CellImpl::delAllConnection ()
{
    for( int i = 0; i < _numConnections; ++i ) {
        Cell* otherCell(_connectingCells[i]);
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

int CellImpl::getNumConnections () const
{
    return _numConnections;
}

void CellImpl::setNumConnections (int num)
{
    _numConnections = num;
}

int CellImpl::getMaxConnections () const
{
    return _maxConnections;
}

void CellImpl::setMaxConnections (int maxConnections)
{

    //new array
    Cell** newArray = new Cell*[maxConnections];
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


Cell* CellImpl::getConnection (int i) const
{
    return _connectingCells[i];
}

void CellImpl::setConnection (int i, Cell* cell)
{
    _connectingCells[i] = cell;
}

QVector2D CellImpl::calcNormal (QVector2D outerSpace) const
{
    if( _numConnections < 2 ) {
        return outerSpace.normalized();
    }

    //find adjacent cells to the outerSpace vector
    outerSpace.normalize();
    Cell* minCell = nullptr;
    Cell* maxCell = nullptr;
    QVector2D minVector;
    QVector2D maxVector;
    qreal minH(0.0);
    qreal maxH(0.0);

    for(int i = 0; i < _numConnections; ++i) {

        //calculate h (angular distance from outerSpace vector)
        //QVector2D u = (transform.map(_connectingCells[i]->getRelPosition())-transform.map(_relPos)).normalized(); OLD
        QVector2D u = (_connectingCells[i]->calcPosition()- calcPosition()).normalized();
        qreal h = QVector2D::dotProduct(outerSpace, u);
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
        //return transform.map(_relPos)-transform.map(minCell->getRelPosition()); OLD
        return calcPosition()-minCell->calcPosition();
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

void CellImpl::activatingNewTokens ()
{
    _tokenStackPointer = _newTokenStackPointer;
    for( int i = 0; i < _newTokenStackPointer; ++i ) {
        _tokenStack[i] = _newTokenStack[i];
    }
    _newTokenStackPointer = 0;
}

const quint64& CellImpl::getId () const
{
    return _id;
}

void CellImpl::setId (quint64 id)
{
    _id = id;
}

const quint64& CellImpl::getTag () const
{
    return _tag;
}

void CellImpl::setTag (quint64 tag)
{
    _tag = tag;
}

int CellImpl::getNumToken (bool newTokenStackPointer /* = false*/) const
{
    if( newTokenStackPointer )
        return _newTokenStackPointer;
    else
        return _tokenStackPointer;
}

Token* CellImpl::getToken (int i) const
{
    return _tokenStack[i];
}

void CellImpl::setToken (int i, Token* token)
{
    _tokenStack[i] = token;
}

void CellImpl::addToken (Token* token, ActivateToken act, UpdateTokenBranchNumber update)
{
    if( update == UpdateTokenBranchNumber::YES )
        token->setTokenAccessNumber(_tokenBranchNumber);
    if( act == ActivateToken::NOW )
        _tokenStack[_tokenStackPointer++] = token;
    else
        _newTokenStack[_newTokenStackPointer++] = token;
}

void CellImpl::delAllTokens ()
{
    for( int j = 0; j < _tokenStackPointer; ++j )
         delete _tokenStack[j];
    for( int j = 0; j < _newTokenStackPointer; ++j )
         delete _newTokenStack[j];
    _tokenStackPointer = 0;
    _newTokenStackPointer = 0;
}

void CellImpl::setCluster (Cluster* cluster)
{
    _cluster = cluster;
}

Cluster* CellImpl::getCluster() const
{
    return _cluster;
}

QVector2D CellImpl::calcPosition (bool metricCorrection /*= false*/) const
{
    return _cluster->calcPosition(this, metricCorrection);
}

void CellImpl::setAbsPosition (QVector2D pos)
{
    _relPos = _cluster->absToRelPos(pos);
}

void CellImpl::setAbsPositionAndUpdateMap (QVector2D pos)
{
    QVector2D oldPos(calcPosition());
	auto cellMap = _context->getCellMap();
    if(cellMap->getCell(oldPos) == this )
		cellMap->setCell(oldPos, 0);
    _relPos = _cluster->absToRelPos(pos);
    if(cellMap->getCell(pos) == 0 )
		cellMap->setCell(pos, this);
}

QVector2D CellImpl::getRelPosition () const
{
    return _relPos;
}

void CellImpl::setRelPosition (QVector2D relPos)
{
    _relPos = relPos;
}


int CellImpl::getBranchNumber () const
{
    return _tokenBranchNumber;
}

void CellImpl::setBranchNumber (int i)
{
    _tokenBranchNumber = i % _context->getSimulationParameters()->cellMaxTokenBranchNumber;
}

bool CellImpl::isTokenBlocked () const
{
    return _blockToken;
}

void CellImpl::setFlagTokenBlocked (bool block)
{
    _blockToken = block;
}

qreal CellImpl::getEnergy() const
{
    return _energy;
}

qreal CellImpl::getEnergyIncludingTokens() const
{
    qreal energy = _energy;
    for(int i = 0; i < _tokenStackPointer; ++i)
        energy += _tokenStack[i]->getEnergy();
    for(int i = 0; i < _newTokenStackPointer; ++i)
        energy += _newTokenStack[i]->getEnergy();
    return energy;
}

void CellImpl::setEnergy (qreal i)
{
    _energy = i;
}

QVector2D CellImpl::getVelocity () const
{
    return _vel;
}

void CellImpl::setVelocity (QVector2D vel)
{
    _vel = vel;
}

int CellImpl::getProtectionCounter () const
{
    return _protectionCounter;
}

void CellImpl::setProtectionCounter (int counter)
{
    _protectionCounter = counter;
}

bool CellImpl::isToBeKilled() const
{
    return _toBeKilled;
}

void CellImpl::setToBeKilled (bool toBeKilled)
{
    _toBeKilled = toBeKilled;
}

CellMetadata CellImpl::getMetadata() const
{
	return _metadata;
}

void CellImpl::setMetadata(CellMetadata metadata)
{
	_metadata = metadata;
}

Token* CellImpl::takeTokenFromStack ()
{
    if( _tokenStackPointer == 0 )
        return 0;
    else {
        return _tokenStack[--_tokenStackPointer];
    }
}

void CellImpl::mutationByChance()
{
	if (_context->getNumberGenerator()->getRandomReal() < _context->getSimulationParameters()->cellMutationProb) {
		_features->mutate();
	}
}

void CellImpl::serializePrimitives (QDataStream& stream) const
{
    //token
    /*stream << _tokenStackPointer;
    for( int i = 0; i < _tokenStackPointer; ++i) {
        _tokenStack[i]->serialize(stream);
    }*/
    stream << static_cast<quint32>(_tokenStackPointer);

    //remaining data
    stream << _toBeKilled << _tag << _id << _protectionCounter << _relPos
           << _energy << _maxConnections << _numConnections;

    //connecting cells
    /*for( int i = 0; i < _numConnections; ++i) {
        stream << _connectingCells[i]->getId();
    }
	*/
    //remaining data
    stream << _tokenBranchNumber << _blockToken << _vel;
}

void CellImpl::deserializePrimitives(QDataStream& stream)
{

	//token stack
    quint32 tokenStackPointer;
    stream >> tokenStackPointer;
    _tokenStackPointer = static_cast<quint32>(tokenStackPointer);
	auto parameters = _context->getSimulationParameters();
    if (_tokenStackPointer > parameters->cellMaxToken)
        _tokenStackPointer = parameters->cellMaxToken;
    _newTokenStackPointer = 0;

	//remaining data
    int numConnections;
	stream >> _toBeKilled >> _tag >> _id >> _protectionCounter >> _relPos
		>> _energy >> _maxConnections >> numConnections;

	//connecting cells
	_connectingCells = 0;
	resetConnections(_maxConnections);
	_numConnections = numConnections;

	//remaining data
	stream >> _tokenBranchNumber >> _blockToken >> _vel;
}


