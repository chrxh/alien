#pragma once

#include <QtGlobal>
#include <QVector>
#include <QVector2D>

#include "Model/Api/ChangeDescriptions.h"
#include "Definitions.h"

class Cell
{
public:

	Cell(uint64_t id, qreal energy, UnitContext* context, int maxConnections, int tokenBranchNumber);

    ~Cell();

	virtual void setContext(UnitContext* context);

	virtual CellDescription getDescription(ResolveDescription const& resolveDescription) const;
	virtual void applyChangeDescription(CellChangeDescription const& change);

    void registerFeatures (CellFeatureChain* features);
    CellFeatureChain* getFeatures () const;
    void removeFeatures ();

    bool connectable (Cell* otherCell) const;
    bool isConnectedTo (Cell* otherCell) const;
    void resetConnections (int maxConnections);
    void newConnection (Cell* otherCell);
    void delConnection (Cell* otherCell);
    void delAllConnection ();
    int getNumConnections () const;
    void setNumConnections (int num);
    int getMaxConnections () const;
    void setMaxConnections (int maxConnections);
    Cell* getConnection (int i) const;
    void setConnection (int i, Cell* cell);
    QVector2D calcNormal (QVector2D outerSpace) const;

    void activatingNewTokens ();
    const quint64& getId () const;
    void setId (quint64 id);
    const quint64& getTag () const;
    void setTag (quint64 tag);
    int getNumToken (bool newTokenStackPointer = false) const;
    Token* getToken (int i) const;
    void setToken (int i, Token* token);
	enum class ActivateToken { Now, Later };
	enum class UpdateTokenBranchNumber { Yes, No };
	void addToken(Token* token, ActivateToken act = ActivateToken::Now, UpdateTokenBranchNumber update = UpdateTokenBranchNumber::Yes);
    void delAllTokens ();
    Token* takeTokenFromStack ();
	void mutationByChance();

    void setCluster (Cluster* cluster);
    Cluster* getCluster () const;
    QVector2D calcPosition (bool metricCorrection = false) const;
    void setAbsPosition (QVector2D pos);
    void setAbsPositionAndUpdateMap (QVector2D pos);
    QVector2D getRelPosition () const;
    void setRelPosition (QVector2D relPos);

    int getBranchNumber () const;
    void setBranchNumber (int i);
    bool isTokenBlocked () const;
    void setFlagTokenBlocked (bool block);
    qreal getEnergy() const;
    qreal getEnergyIncludingTokens() const;
    void setEnergy (qreal i);

    QVector2D getVelocity () const;
    void setVelocity (QVector2D vel);
    int getProtectionCounter () const;
    void setProtectionCounter (int counter);
    bool isToBeKilled() const;
    void setToBeKilled (bool toBeKilled);

	CellMetadata getMetadata() const;
	void setMetadata(CellMetadata metadata);

private:
    friend class Cluster;

	UnitContext* _context = nullptr;
	CellFeatureChain* _features = nullptr;

    QVector<Token*> _tokenStack;
    QVector<Token*> _newTokenStack;
    int _tokenStackPointer = 0;
    int _newTokenStackPointer = 0;

    bool _toBeKilled = false;
    quint64 _tag = 0;
    quint64 _id = 0;
    int _protectionCounter = 0;
    QVector2D _relPos;
    Cluster* _cluster = nullptr;
    qreal _energy = 0.0;

    int _maxConnections = 0;
    int _numConnections = 0;
    Cell** _connectingCells = nullptr;

    int _tokenBranchNumber = 0;
    bool _blockToken = false;

    QVector2D _vel;

	CellMetadata _metadata;
};

