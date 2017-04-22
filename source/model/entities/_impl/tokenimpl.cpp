#include "global/numbergenerator.h"
#include "model/modelsettings.h"
#include "model/simulationparameters.h"
#include "model/simulationcontext.h"

#include "tokenimpl.h"

TokenImpl::TokenImpl(SimulationContext * context)
	: _context(context)
{
	_memory = QByteArray(context->getSimulationParameters()->cellFunctionComputerTokenMemorySize, 0);
}

TokenImpl::TokenImpl(SimulationContext* context, qreal energy, bool randomData)
	: TokenImpl(context)
{
	_energy = energy;
	if (randomData) {
		for (int i = 0; i < context->getSimulationParameters()->cellFunctionComputerTokenMemorySize; ++i)
			_memory[i] = NumberGenerator::getInstance().random(256);
	}
}

TokenImpl::TokenImpl(SimulationContext* context, qreal energy, QByteArray const& memory_)
	: _energy(energy), _context(context)
{
	_memory = memory_.left(context->getSimulationParameters()->cellFunctionComputerTokenMemorySize);
}

TokenImpl* TokenImpl::duplicate() const
{
	TokenImpl* newToken(new TokenImpl(_context));
	for (int i = 0; i < _context->getSimulationParameters()->cellFunctionComputerTokenMemorySize; ++i)
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
	auto memSize = _context->getSimulationParameters()->cellFunctionComputerTokenMemorySize;
	_memory = _memory.left(memSize);
	_memory.resize(memSize);
}
