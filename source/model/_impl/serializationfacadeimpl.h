#ifndef SERIALIZATIONFACADEIMPL_H
#define SERIALIZATIONFACADEIMPL_H

#include "model/serializationfacade.h"

class SerializationFacadeImpl : public SerializationFacade
{
public:
	SerializationFacadeImpl();
	~SerializationFacadeImpl() = default;

	void serializeSimulationContext(SimulationContext* context, QDataStream& stream) override;
	SimulationContext* deserializeSimulationContext(QDataStream& stream) override;

	void serializeCellCluster(CellCluster* cluster, QDataStream& stream) override;
	CellCluster* deserializeCellCluster(QDataStream& stream, QMap< quint64, quint64 >& oldNewClusterIdMap
		, QMap< quint64, quint64 >& oldNewCellIdMap, QMap< quint64, Cell* >& oldIdCellMap, SimulationContext* context) override;

	void serializeFeaturedCell(Cell* cell, QDataStream& stream) override;
    Cell* deserializeFeaturedCell(QDataStream& stream, QMap< quint64, QList< quint64 > >& connectingCells, SimulationContext* context) override;
    Cell* deserializeFeaturedCell(QDataStream& stream, SimulationContext* context) override;

private:
    void serializeToken(Token* token, QDataStream& stream) override;
    Token* deserializeToken(QDataStream& stream, SimulationContext* context) override;
};

#endif //SERIALIZATIONFACADEIMPL_H
