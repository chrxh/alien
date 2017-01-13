#ifndef CELLIMPL_H
#define CELLIMPL_H

#include "model/entities/cell.h"

#include <QtGlobal>
#include <QVector3D>
#include <QVector>

class CellImpl : public Cell
{
public:

    CellImpl (SimulationContext* context);
    CellImpl (qreal energy, SimulationContext* context, int maxConnections
        , int tokenAccessNumber, QVector3D relPos);

    ~CellImpl();

    void registerFeatures (CellFeature* features) override;
    CellFeature* getFeatures () const override;
    void removeFeatures () override;

    bool connectable (Cell* otherCell) const override;
    bool isConnectedTo (Cell* otherCell) const override;
    void resetConnections (int maxConnections) override;
    void newConnection (Cell* otherCell) override;
    void delConnection (Cell* otherCell) override;
    void delAllConnection () override;
    int getNumConnections () const override;
    void setNumConnections (int num) override;
    int getMaxConnections () const override;
    void setMaxConnections (int maxConnections) override;
    Cell* getConnection (int i) const override;
    void setConnection (int i, Cell* cell) override;
    QVector3D calcNormal (QVector3D outerSpace) const override;

    void activatingNewTokens () override;
    const quint64& getId () const override;
    void setId (quint64 id) override;
    const quint64& getTag () const override;
    void setTag (quint64 tag) override;
    int getNumToken (bool newTokenStackPointer = false) const override;
    Token* getToken (int i) const override;
    void setToken (int i, Token* token) override;
    void addToken (Token* token, ACTIVATE_TOKEN act, UPDATE_TOKEN_ACCESS_NUMBER update) override;
    void delAllTokens () override;
    Token* takeTokenFromStack () override;

    void setCluster (CellCluster* cluster) override;
    CellCluster* getCluster () const override;
    QVector3D calcPosition (bool topologyCorrection = false) const override;
    void setAbsPosition (QVector3D pos) override;
    void setAbsPositionAndUpdateMap (QVector3D pos) override;
    QVector3D getRelPos () const override;
    void setRelPos (QVector3D relPos) override;

    int getTokenAccessNumber () const override;
    void setTokenAccessNumber (int i) override;
    bool isTokenBlocked () const override;
    void setTokenBlocked (bool block) override;
    qreal getEnergy() const override;
    qreal getEnergyIncludingTokens() const override;
    void setEnergy (qreal i) override;

    QVector3D getVel () const override;
    void setVel (QVector3D vel) override;
    int getProtectionCounter () const override;
    void setProtectionCounter (int counter) override;
    bool isToBeKilled() const override;
    void setToBeKilled (bool toBeKilled) override;

	CellMetadata getMetadata() const override;
	void setMetadata(CellMetadata metadata) override;

    void serializePrimitives (QDataStream& stream) const override;
    void deserializePrimitives(QDataStream& stream) override;

private:
    friend class CellCluster;

    CellMap* _cellMap = nullptr;
    CellFeature* _features = nullptr;

    QVector< Token* > _tokenStack;
    QVector< Token* > _newTokenStack;
    int _tokenStackPointer = 0;
    int _newTokenStackPointer = 0;

    bool _toBeKilled = false;
    quint64 _tag = 0;
    quint64 _id = 0;
    int _protectionCounter = 0;
    QVector3D _relPos;
    CellCluster* _cluster = nullptr;
    qreal _energy = 0.0;

    int _maxConnections = 0;
    int _numConnections = 0;
    Cell** _connectingCells = nullptr;

    int _tokenAccessNumber = 0;
    bool _blockToken = false;

    QVector3D _vel;

	CellMetadata _metadata;
};

#endif // CELLIMPL_H


