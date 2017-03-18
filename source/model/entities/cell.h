#ifndef CELL_H
#define CELL_H

#include <QVector3D>

#include "model/definitions.h"

class Cell
{
public:
    virtual ~Cell() {}

    virtual void registerFeatures (CellFeature* features) = 0;
    virtual CellFeature* getFeatures () const = 0;
    virtual void removeFeatures () = 0;

    virtual bool connectable (Cell* otherCell) const = 0;
    virtual bool isConnectedTo (Cell* otherCell) const = 0;
    virtual void resetConnections (int maxConnections) = 0;
    virtual void newConnection (Cell* otherCell) = 0;
    virtual void delConnection (Cell* otherCell) = 0;
    virtual void delAllConnection () = 0;
    virtual int getNumConnections () const = 0;
    virtual void setNumConnections (int num) = 0;
    virtual int getMaxConnections () const = 0;
    virtual void setMaxConnections (int maxConnections) = 0;
    virtual Cell* getConnection (int i) const = 0;
    virtual void setConnection (int i, Cell* cell) = 0;
    virtual QVector3D calcNormal (QVector3D outerSpace) const = 0;

    virtual void activatingNewTokens () = 0;
    virtual const quint64& getId () const = 0;
    virtual void setId (quint64 id) = 0;
    virtual const quint64& getTag () const = 0;
    virtual void setTag (quint64 tag) = 0;
    virtual int getNumToken (bool newTokenStackPointer = false) const = 0;
    virtual Token* getToken (int i) const = 0;
    virtual void setToken (int i, Token* token) = 0;
    enum class ActivateToken { NOW, LATER };
    enum class UpdateTokenAccessNumber { YES, NO };
    virtual void addToken (Token* token, ActivateToken act = ActivateToken::NOW
        , UpdateTokenAccessNumber update = UpdateTokenAccessNumber::YES) = 0;
    virtual void delAllTokens () = 0;

    virtual void setCluster (CellCluster* cluster) = 0;
    virtual CellCluster* getCluster () const = 0;
    virtual QVector3D calcPosition (bool topologyCorrection = false) const = 0;
    virtual void setAbsPosition (QVector3D pos) = 0;
    virtual void setAbsPositionAndUpdateMap (QVector3D pos) = 0;
    virtual QVector3D getRelPos () const = 0;
    virtual void setRelPos (QVector3D relPos) = 0;

    virtual int getTokenAccessNumber () const = 0;
    virtual void setTokenAccessNumber (int i) = 0;
    virtual bool isTokenBlocked () const = 0;
    virtual void setTokenBlocked (bool block) = 0;
    virtual qreal getEnergy() const = 0;
    virtual qreal getEnergyIncludingTokens() const = 0;
    virtual void setEnergy (qreal i) = 0;

    virtual QVector3D getVel () const = 0;
    virtual void setVel (QVector3D vel) = 0;
    virtual int getProtectionCounter () const = 0;
    virtual void setProtectionCounter (int counter) = 0;
    virtual bool isToBeKilled() const = 0;
    virtual void setToBeKilled (bool toBeKilled) = 0;
    virtual Token* takeTokenFromStack () = 0;

	virtual CellMetadata getMetadata() const = 0;
	virtual void setMetadata(CellMetadata metadata) = 0;

    virtual void serializePrimitives (QDataStream& stream) const = 0;
    virtual void deserializePrimitives(QDataStream& stream) = 0;
};

#endif // CELL_H
