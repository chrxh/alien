#ifndef CELLIMPL_H
#define CELLIMPL_H

#include "model/entities/Cell.h"

#include <QtGlobal>
#include <QVector2D>
#include <QVector>

class CellImpl
	: public Cell
{
public:

    CellImpl (UnitContext* context);
    CellImpl (qreal energy, UnitContext* context, int maxConnections
        , int tokenAccessNumber, QVector2D relPos);

    ~CellImpl();

	virtual void setContext(UnitContext* context) override;

	virtual CellDescription getDescription() const override;

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
    QVector2D calcNormal (QVector2D outerSpace) const override;

    void activatingNewTokens () override;
    const quint64& getId () const override;
    void setId (quint64 id) override;
    const quint64& getTag () const override;
    void setTag (quint64 tag) override;
    int getNumToken (bool newTokenStackPointer = false) const override;
    Token* getToken (int i) const override;
    void setToken (int i, Token* token) override;
	void addToken(Token* token, ActivateToken act = ActivateToken::NOW, UpdateTokenAccessNumber update = UpdateTokenAccessNumber::YES) override;
    void delAllTokens () override;
    Token* takeTokenFromStack () override;
	void mutationByChance() override;

    void setCluster (CellCluster* cluster) override;
    CellCluster* getCluster () const override;
    QVector2D calcPosition (bool metricCorrection = false) const override;
    void setAbsPosition (QVector2D pos) override;
    void setAbsPositionAndUpdateMap (QVector2D pos) override;
    QVector2D getRelPosition () const override;
    void setRelPosition (QVector2D relPos) override;

    int getBranchNumber () const override;
    void setBranchNumber (int i) override;
    bool isTokenBlocked () const override;
    void setFlagTokenBlocked (bool block) override;
    qreal getEnergy() const override;
    qreal getEnergyIncludingTokens() const override;
    void setEnergy (qreal i) override;

    QVector2D getVelocity () const override;
    void setVelocity (QVector2D vel) override;
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

	UnitContext* _context = nullptr;
	CellFeature* _features = nullptr;

    QVector<Token*> _tokenStack;
    QVector<Token*> _newTokenStack;
    int _tokenStackPointer = 0;
    int _newTokenStackPointer = 0;

    bool _toBeKilled = false;
    quint64 _tag = 0;
    quint64 _id = 0;
    int _protectionCounter = 0;
    QVector2D _relPos;
    CellCluster* _cluster = nullptr;
    qreal _energy = 0.0;

    int _maxConnections = 0;
    int _numConnections = 0;
    Cell** _connectingCells = nullptr;

    int _tokenAccessNumber = 0;
    bool _blockToken = false;

    QVector2D _vel;

	CellMetadata _metadata;
};

#endif // CELLIMPL_H


