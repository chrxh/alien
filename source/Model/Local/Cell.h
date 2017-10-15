#pragma once

#include <QVector2D>

#include "Model/Api/Definitions.h"
#include "Model/Api/ChangeDescriptions.h"

class Cell
{
public:
	virtual ~Cell() = default;

	virtual void setContext(UnitContext* context) = 0;

	virtual CellDescription getDescription(ResolveDescription const& resolveDescription) const = 0;
	virtual void applyChangeDescription(CellChangeDescription const& change) = 0;

    virtual void registerFeatures (CellFeatureChain* features) = 0;
    virtual CellFeatureChain* getFeatures () const = 0;
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
    virtual QVector2D calcNormal (QVector2D outerSpace) const = 0;

    virtual void activatingNewTokens () = 0;
    virtual const quint64& getId () const = 0;
    virtual void setId (quint64 id) = 0;
    virtual const quint64& getTag () const = 0;
    virtual void setTag (quint64 tag) = 0;
    virtual int getNumToken (bool newTokenStackPointer = false) const = 0;
    virtual Token* getToken (int i) const = 0;
    virtual void setToken (int i, Token* token) = 0;
    enum class ActivateToken { NOW, LATER };
    enum class UpdateTokenBranchNumber { YES, NO };
    virtual void addToken (Token* token, ActivateToken act = ActivateToken::NOW, UpdateTokenBranchNumber update = UpdateTokenBranchNumber::YES) = 0;
    virtual void delAllTokens () = 0;

    virtual void setCluster (Cluster* cluster) = 0;
    virtual Cluster* getCluster () const = 0;
    virtual QVector2D calcPosition (bool metricCorrection = false) const = 0;
    virtual void setAbsPosition (QVector2D pos) = 0;
    virtual void setAbsPositionAndUpdateMap (QVector2D pos) = 0;
    virtual QVector2D getRelPosition () const = 0;
    virtual void setRelPosition (QVector2D relPos) = 0;

    virtual int getBranchNumber () const = 0;
    virtual void setBranchNumber (int i) = 0;
    virtual bool isTokenBlocked () const = 0;
    virtual void setFlagTokenBlocked (bool block) = 0;
    virtual qreal getEnergy() const = 0;
    virtual qreal getEnergyIncludingTokens() const = 0;
    virtual void setEnergy (qreal i) = 0;

    virtual QVector2D getVelocity () const = 0;
    virtual void setVelocity (QVector2D vel) = 0;
    virtual int getProtectionCounter () const = 0;
    virtual void setProtectionCounter (int counter) = 0;
    virtual bool isToBeKilled() const = 0;
    virtual void setToBeKilled (bool toBeKilled) = 0;
    virtual Token* takeTokenFromStack () = 0;
	virtual void mutationByChance() = 0;

	virtual CellMetadata getMetadata() const = 0;
	virtual void setMetadata(CellMetadata metadata) = 0;

    virtual void serializePrimitives (QDataStream& stream) const = 0;
    virtual void deserializePrimitives(QDataStream& stream) = 0;
};

