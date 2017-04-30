#include "global/numbergenerator.h"
#include "model/modelsettings.h"
#include "model/context/simulationparameters.h"
#include "model/context/simulationunitcontext.h"

#include "tokenimpl.h"

TokenImpl::TokenImpl(SimulationUnitContext * context)
	: _context(context)
{
	_memory = QByteArray(context->getSimulationParameters()->tokenMemorySize, 0);
}

TokenImpl::TokenImpl(SimulationUnitContext* context, qreal energy, bool randomData)
	: TokenImpl(context)
{
	_energy = energy;
	if (randomData) {
		for (int i = 0; i < context->getSimulationParameters()->tokenMemorySize; ++i)
			_memory[i] = NumberGenerator::getInstance().random(256);
	}
}

TokenImpl::TokenImpl(SimulationUnitContext* context, qreal energy, QByteArray const& memory_)
	: _energy(energy), _context(context)
{
	_memory = memory_.left(context->getSimulationParameters()->tokenMemorySize);
}

TokenImpl* TokenImpl::duplicate() const
{
	TokenImpl* newToken(new TokenImpl(_context));
	for (int i = 0; i < _context->getSimulationParameters()->tokenMemorySize; ++i)
		newToken->_memory[i] = _memory[i];
	newToken->_energy = _energy;

	return newToken;
}

int TokenImpl::getTokenAccessNumber() const
{
	return _memory[0] % _context->getSimulationParameters()->cellMaxTokenBranchNumber;
}

void TokenImpl::setTokenAccessNumber(int i)
{
	_memory[0] = i;
}

void TokenImpl::setEnergy(qreal energy)
{
	_energy = energy;
}

qreal TokenImpl::getEnergy() const
{
	return _energy;
}

QByteArray & TokenImpl::getMemoryRef() 
{
	return _memory;
}

void TokenImpl::serializePrimitives(QDataStream& stream) const
{
	stream << _memory << _energy;
}

void TokenImpl::deserializePrimitives(QDataStream& stream)
{
	stream >> _memory >> _energy;
	auto memSize = _context->getSimulationParameters()->tokenMemorySize;
	_memory = _memory.left(memSize);
	_memory.resize(memSize);
}
