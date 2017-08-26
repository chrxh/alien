#ifndef SERIALIZATIONFACADEIMPL_H
#define SERIALIZATIONFACADEIMPL_H

#include "Model/SerializationFacade.h"

class SerializationFacadeImpl
	: public SerializationFacade
{
public:
	virtual ~SerializationFacadeImpl() = default;

    void serializeSimulationContext(UnitContext* context, QDataStream& stream) const override;
    void deserializeSimulationContext(UnitContext* prevContext, QDataStream& stream) const override;

	void serializeSimulationParameters(SimulationParameters* parameters, QDataStream& stream) const override;
	SimulationParameters* deserializeSimulationParameters(QDataStream& stream) const override;

	void serializeSymbolTable(SymbolTable* symbolTable, QDataStream& stream) const override;
	SymbolTable* deserializeSymbolTable(QDataStream& stream) const override;

    void serializeCellCluster(Cluster* cluster, QDataStream& stream) const override;
    virtual Cluster* deserializeCellCluster(QDataStream& stream, UnitContext* context) const override;

    void serializeFeaturedCell(Cell* cell, QDataStream& stream) const override;
    Cell* deserializeFeaturedCell(QDataStream& stream, UnitContext* context) const override;

    void serializeEnergyParticle(Particle* particle, QDataStream& stream) const override;
    Particle* deserializeEnergyParticle(QDataStream& stream, UnitContext* context) const override;

    void serializeToken(Token* token, QDataStream& stream) const override;
    Token* deserializeToken(QDataStream& stream, UnitContext* context) const override;

private:
    Cluster* deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
        , QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap
        , UnitContext* context) const;

    Particle* deserializeEnergyParticle(QDataStream& stream
        , QMap< quint64, Particle* >& oldIdEnergyMap, UnitContext* context) const;

    Cell* deserializeFeaturedCell(QDataStream& stream, QMap< quint64
        , QList< quint64 > >& connectingCells, UnitContext* context) const;
};

#endif //SERIALIZATIONFACADEIMPL_H
