#pragma once

#include "Model/Local/Token.h"

class TokenImpl
	: public Token
{
public:
	TokenImpl(UnitContext* context);
	TokenImpl(UnitContext* context, qreal energy, QByteArray const& memory);

	virtual void setContext(UnitContext* context) override;

	TokenImpl* duplicate() const override;

	virtual TokenDescription getDescription() const override;

	int getTokenAccessNumber() const override;        //from memory[0]
	void setTokenAccessNumber(int i) override;

	void setEnergy(qreal energy) override;
	qreal getEnergy() const override;

	QByteArray& getMemoryRef() override;

private:
	UnitContext* _context = nullptr;

	QByteArray _memory;
	qreal _energy = 0.0;
};

