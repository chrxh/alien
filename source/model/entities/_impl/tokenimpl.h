#ifndef TOKENIMPL_H
#define TOKENIMPL_H

#include "model/entities/token.h"

class TokenImpl
	: public Token
{
public:
	TokenImpl(SimulationUnitContext* context);
	TokenImpl(SimulationUnitContext* context, qreal energy, bool randomData = false);
	TokenImpl(SimulationUnitContext* context, qreal energy, QByteArray const& memory);

	TokenImpl* duplicate() const override;
	int getTokenAccessNumber() const override;        //from memory[0]
	void setTokenAccessNumber(int i) override;

	void setEnergy(qreal energy) override;
	qreal getEnergy() const override;

	QByteArray& getMemoryRef() override;

	void serializePrimitives(QDataStream& stream) const override;
	void deserializePrimitives(QDataStream& stream) override;

private:
	SimulationUnitContext* _context = nullptr;

	QByteArray _memory;
	qreal _energy = 0.0;
};

#endif // TOKENIMPL_H
