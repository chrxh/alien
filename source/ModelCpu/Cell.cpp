#include "Base/ServiceLocator.h"
#include "Base/NumberGenerator.h"

#include "ModelBasic/SimulationParameters.h"
#include "ModelBasic/Physics.h"
#include "Cluster.h"
#include "Token.h"
#include "CellFeatureChain.h"
#include "UnitContext.h"
#include "CellMap.h"
#include "EntityFactory.h"
#include "CellFeatureFactory.h"

#include "Cell.h"

Cell::Cell (uint64_t id, qreal energy, UnitContext* context, int maxConnections, int tokenBranchNumber)
	: _id(id)
	, _context(context)
	, _tokenStack(context->getSimulationParameters().cellMaxToken)
	, _newTokenStack(context->getSimulationParameters().cellMaxToken)
{
	_energy = energy;
    _tokenBranchNumber = tokenBranchNumber;
    resetConnections(maxConnections);
}

Cell::~Cell()
{
    for(int i = 0; i < _tokenStackPointer; ++i )
        delete _tokenStack[i];
    for(int i = 0; i < _newTokenStackPointer; ++i )
        delete _newTokenStack[i];
    if( _maxConnections > 0 )
        delete _connectingCells;
    delete _features;
}

void Cell::setContext(UnitContext * context)
{
	_context = context;
	for (int i = 0; i < _tokenStackPointer; ++i) {
		_tokenStack[i]->setContext(context);
	}
	if (_features) {
		_features->setContext(context);
	}
}

CellDescription Cell::getDescription(ResolveDescription const& resolveDescription) const
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
	for (int i = 0; i < getNumToken(); ++i) {
		result.addToken(getToken(i)->getDescription());
	}
	return result;
}

void Cell::applyChangeDescription(CellChangeDescription const & change)
{
	if (change.energy) {
		setEnergy(*change.energy);
	}
	if (change.maxConnections) {
		setMaxConnections(*change.maxConnections);
	}
	if (change.tokenBlocked) {
		setFlagTokenBlocked(*change.tokenBlocked);
	}
	if (change.tokenBranchNumber) {
		setBranchNumber(*change.tokenBranchNumber);
	}
	if (change.metadata) {
		setMetadata(*change.metadata);
	}
	if (change.cellFeatures) {
		delete _features;
		CellFeatureFactory* factory = ServiceLocator::getInstance().getService<CellFeatureFactory>();
		auto features = factory->build(*change.cellFeatures, _context);
		registerFeatures(features);
	}
	if (change.tokens) {
		delAllTokens();
		EntityFactory* factory = ServiceLocator::getInstance().getService<EntityFactory>();
		for (auto const& tokenDesc : *change.tokens) {
			addToken(factory->build(tokenDesc, _context));
		}
	}
}

void Cell::registerFeatures (CellFeatureChain* features)
{
    _features = features;
}

CellFeatureChain* Cell::getFeatures () const
{
    return _features;
}

void Cell::removeFeatures ()
{
    delete _features;
    _features = nullptr;
}

bool Cell::connectable (Cell* otherCell) const
{
    return (_numConnections < _maxConnections) && (otherCell->getNumConnections() < otherCell->getMaxConnections());
}

bool Cell::isConnectedTo (Cell* otherCell) const
{
    for( int i = 0; i < _numConnections; ++i )
        if( _connectingCells[i] == otherCell )
            return true;
    return false;
}

void Cell::resetConnections (int maxConnections)
{
    //delete old array
    delete _connectingCells;

    //set up new array
    _maxConnections = maxConnections;
    _numConnections = 0;
    _connectingCells = new Cell*[maxConnections];
}

void Cell::newConnection (Cell* otherCell)
{
    _connectingCells[_numConnections] = otherCell;
    _numConnections++;
    otherCell->setConnection(otherCell->getNumConnections(), this);
    otherCell->setNumConnections(otherCell->getNumConnections()+1);
}

void Cell::delConnection (Cell* otherCell)
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

void Cell::delAllConnection ()
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

int Cell::getNumConnections () const
{
    return _numConnections;
}

void Cell::setNumConnections (int num)
{
    _numConnections = num;
}

int Cell::getMaxConnections () const
{
    return _maxConnections;
}

void Cell::setMaxConnections (int maxConnections)
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


Cell* Cell::getConnection (int i) const
{
    return _connectingCells[i];
}

void Cell::setConnection (int i, Cell* cell)
{
    _connectingCells[i] = cell;
}

QVector2D Cell::calcNormal (QVector2D outerSpace) const
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

void Cell::activatingNewTokens ()
{
    _tokenStackPointer = _newTokenStackPointer;
    for( int i = 0; i < _newTokenStackPointer; ++i ) {
        _tokenStack[i] = _newTokenStack[i];
    }
    _newTokenStackPointer = 0;
}

const quint64& Cell::getId () const
{
    return _id;
}

void Cell::setId (quint64 id)
{
    _id = id;
}

const quint64& Cell::getTag () const
{
    return _tag;
}

void Cell::setTag (quint64 tag)
{
    _tag = tag;
}

int Cell::getNumToken (bool newTokenStackPointer /* = false*/) const
{
    if( newTokenStackPointer )
        return _newTokenStackPointer;
    else
        return _tokenStackPointer;
}

Token* Cell::getToken (int i) const
{
    return _tokenStack[i];
}

void Cell::setToken (int i, Token* token)
{
    _tokenStack[i] = token;
}

void Cell::addToken (Token* token, ActivateToken act, UpdateTokenBranchNumber update)
{
    if( update == UpdateTokenBranchNumber::Yes )
        token->setTokenAccessNumber(_tokenBranchNumber);
    if( act == ActivateToken::Now )
        _tokenStack[_tokenStackPointer++] = token;
    else
        _newTokenStack[_newTokenStackPointer++] = token;
}

void Cell::delAllTokens ()
{
    for( int j = 0; j < _tokenStackPointer; ++j )
         delete _tokenStack[j];
    for( int j = 0; j < _newTokenStackPointer; ++j )
         delete _newTokenStack[j];
    _tokenStackPointer = 0;
    _newTokenStackPointer = 0;
}

void Cell::setCluster (Cluster* cluster)
{
    _cluster = cluster;
	setContext(cluster->getContext());
}

Cluster* Cell::getCluster() const
{
    return _cluster;
}

QVector2D Cell::calcPosition (bool metricCorrection /*= false*/) const
{
    return _cluster->calcPosition(this, metricCorrection);
}

void Cell::setAbsPosition (QVector2D pos)
{
    _relPos = _cluster->absToRelPos(pos);
}

void Cell::setAbsPositionAndUpdateMap (QVector2D pos)
{
    QVector2D oldPos(calcPosition());
	auto cellMap = _context->getCellMap();
    if(cellMap->getCell(oldPos) == this )
		cellMap->setCell(oldPos, 0);
    _relPos = _cluster->absToRelPos(pos);
    if(cellMap->getCell(pos) == 0 )
		cellMap->setCell(pos, this);
}

QVector2D Cell::getRelPosition () const
{
    return _relPos;
}

void Cell::setRelPosition (QVector2D relPos)
{
    _relPos = relPos;
}


int Cell::getBranchNumber () const
{
    return _tokenBranchNumber;
}

void Cell::setBranchNumber (int i)
{
    _tokenBranchNumber = i % _context->getSimulationParameters().cellMaxTokenBranchNumber;
}

bool Cell::isTokenBlocked () const
{
    return _blockToken;
}

void Cell::setFlagTokenBlocked (bool block)
{
    _blockToken = block;
}

qreal Cell::getEnergy() const
{
    return _energy;
}

qreal Cell::getEnergyIncludingTokens() const
{
    qreal energy = _energy;
    for(int i = 0; i < _tokenStackPointer; ++i)
        energy += _tokenStack[i]->getEnergy();
    for(int i = 0; i < _newTokenStackPointer; ++i)
        energy += _newTokenStack[i]->getEnergy();
    return energy;
}

void Cell::setEnergy (qreal i)
{
    _energy = i;
}

QVector2D Cell::getVelocity () const
{
    return _vel;
}

void Cell::setVelocity (QVector2D vel)
{
    _vel = vel;
}

int Cell::getProtectionCounter () const
{
    return _protectionCounter;
}

void Cell::setProtectionCounter (int counter)
{
    _protectionCounter = counter;
}

bool Cell::isToBeKilled() const
{
    return _toBeKilled;
}

void Cell::setToBeKilled (bool toBeKilled)
{
    _toBeKilled = toBeKilled;
}

CellMetadata Cell::getMetadata() const
{
	return _metadata;
}

void Cell::setMetadata(CellMetadata metadata)
{
	_metadata = metadata;
}

Token* Cell::takeTokenFromStack ()
{
    if( _tokenStackPointer == 0 )
        return 0;
    else {
        return _tokenStack[--_tokenStackPointer];
    }
}

void Cell::mutationByChance()
{
	if (_context->getNumberGenerator()->getRandomReal() < _context->getSimulationParameters().cellFunctionConstructorCellDataMutationProb) {
		_features->mutate();
	}
}

